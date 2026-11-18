"""Hair of any length for the viewer: an HG hairstyle's strands grown (or trimmed) and combed, as hair cards.

Used by export_model.py when the character has a hair length or waves set (edit_model.py: hair_length_cm,
hair_wave; stored on the rig as studio_hair_length / studio_hair_wave). HG's own strand hair can't be lengthened
by script in this Blender (its strand points read as zeros without a window), so this works from the strands
HG's hair-card step also uses (the particle systems converted to lines):
1. strands: each line ordered root (near the head) to tip; about 2000 kept.
2. grow: each strand extended along its own direction, bending down with gravity, to the set length from the root,
   pushed out of the head and body; or trimmed if the length is shorter. Waves (0-1): a sideways wave growing
   toward the ends.
3. cards: a tapered ribbon along each strand, facing out from the head, with a generated strand texture (fine
   streaks in a few clumps, fading at the tip); coloured like the hair. Near the head it follows the head bone,
   lower down the upper back.
"""
import math
import os

import bpy
import numpy as np
from mathutils import Vector
from mathutils.kdtree import KDTree

from HumGen3D.common.context import context_override


def _chains(me, M):
    """Polylines from a mesh of edges (one per strand), in world space."""
    n = len(me.vertices)
    nb = [[] for _ in range(n)]
    for e in me.edges:
        a, b = e.vertices
        nb[a].append(b)
        nb[b].append(a)
    P = np.array([M @ v.co for v in me.vertices])
    seen = np.zeros(n, bool)
    out = []
    for s in range(n):
        if seen[s] or len(nb[s]) != 1:
            continue
        line, cur, prev = [], s, -1
        while True:
            seen[cur] = True
            line.append(cur)
            nxt = [x for x in nb[cur] if x != prev and not seen[x]]
            if not nxt:
                break
            prev, cur = cur, nxt[0]
        if len(line) > 2:
            out.append(P[line])
    return out


def strands(human, keep=2000, seed=3):
    """The visible regular-hair systems as strand polylines, root first."""
    body = human.objects.body
    rig = human.objects.rig
    hb = rig.data.bones["head"]
    head_c = np.array(rig.matrix_world @ ((hb.head_local + hb.tail_local) / 2))
    out = []
    for m in [m for m in body.modifiers if m.type == "PARTICLE_SYSTEM" and m.show_viewport
              and not m.particle_system.name.startswith(("Eyebrows", "Eyelash"))]:
        before = set(bpy.data.objects)
        with context_override(bpy.context, body, [body]):
            bpy.ops.object.modifier_convert(modifier=m.name)
        for o in [o for o in bpy.data.objects if o not in before]:
            if o.type == "MESH":
                for line in _chains(o.data, o.matrix_world):
                    if np.linalg.norm(line[-1] - head_c) < np.linalg.norm(line[0] - head_c):
                        line = line[::-1]  # root first
                    out.append(line)
            bpy.data.objects.remove(o, do_unlink=True)
    rng = np.random.default_rng(seed)
    if len(out) > keep:
        out = [out[i] for i in rng.choice(len(out), keep, replace=False)]
    return out, head_c


def _resample(line, n):
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(line, axis=0), axis=1))]
    s = np.linspace(0, seg[-1], n)
    return np.c_[[np.interp(s, seg, line[:, k]) for k in range(3)]].T


def grow(lines, length, wave, head_c, P_body, N_body, head_r=0.105, step=0.015, seed=5):
    """Each strand extended (or trimmed) to `length` metres from its root, bending down, kept out of the body."""
    kd = KDTree(len(P_body))
    for i, p in enumerate(P_body):
        kd.insert(Vector(p), i)
    kd.balance()
    rng = np.random.default_rng(seed)
    down = np.array([0.0, 0.0, -1.0])
    out = []
    for line in lines:
        seg = np.linalg.norm(np.diff(line, axis=0), axis=1)
        have = seg.sum()
        pts = list(line)
        if length and length < have:  # trim
            cum = np.cumsum(seg)
            k = int(np.searchsorted(cum, length)) + 1
            pts = list(line[:k + 1])
        elif length and length > have:
            d = line[-1] - line[-2]
            d /= np.linalg.norm(d) + 1e-9
            total = have
            while total < length:
                d = d * 0.75 + down * 0.25
                d /= np.linalg.norm(d)
                p = pts[-1] + d * step
                # out of the head (a sphere) and the body (nearest body point and its normal)
                v = p - head_c
                r = np.linalg.norm(v)
                if r < head_r and p[2] > head_c[2] - 0.12:
                    p = head_c + v / (r + 1e-9) * head_r
                _, bi, dist = kd.find(Vector(p))
                gap = np.dot(p - P_body[bi], N_body[bi])
                if gap < 0.012:
                    p = p + N_body[bi] * (0.012 - gap)
                d = (p - pts[-1]) / (np.linalg.norm(p - pts[-1]) + 1e-9)
                pts.append(p)
                total += step
        pts = np.array(pts)
        if wave > 0:
            cum = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
            out_dir = pts[0] - head_c
            out_dir[2] = 0
            out_dir /= np.linalg.norm(out_dir) + 1e-9
            side = np.cross(out_dir, down)
            ph = rng.uniform(0, 2 * math.pi)
            amp = 0.022 * wave * np.clip((cum - 0.08) / 0.25, 0, 1)
            pts = pts + (np.sin(cum * 2 * math.pi / 0.09 + ph) * amp)[:, None] * side[None, :] \
                + (np.cos(cum * 2 * math.pi / 0.09 + ph) * amp * 0.6)[:, None] * out_dir[None, :]
        out.append(_resample(pts, 24))
    return out


def strand_texture(path, cols=8, w=96, h=1024, seed=11):
    """A hair-card texture: `cols` clumps side by side, each fine streaks fading at the tip and the edges.
    Saved as RGBA (RGB: shading, darker at the root; A: strands)."""
    rng = np.random.default_rng(seed)
    W = cols * w
    alpha = np.zeros((h, W), np.float32)
    v = np.linspace(0, 1, h)[:, None]
    for c in range(cols):
        for _ in range(70):
            x = c * w + rng.uniform(0.15, 0.85) * w
            wid = rng.uniform(0.6, 1.6)
            end = rng.uniform(0.75, 1.0)
            xs = np.arange(c * w, (c + 1) * w)[None, :]
            drift = (v * rng.uniform(-4, 4))
            streak = np.exp(-((xs - x - drift) ** 2) / (2 * wid ** 2))
            streak *= np.clip((end - v) / 0.15, 0, 1) * rng.uniform(0.5, 1.0)
            alpha[:, c * w:(c + 1) * w] = np.maximum(alpha[:, c * w:(c + 1) * w], streak)
    edge = np.clip(np.minimum(np.arange(W) % w, w - 1 - np.arange(W) % w) / (w * 0.12), 0, 1)[None, :]
    alpha *= edge
    shade = 0.55 + 0.45 * np.clip(v * 1.4, 0, 1)
    rgb = np.repeat(shade, W, 1)
    img = bpy.data.images.new("long_hair_strands", W, h, alpha=True, float_buffer=False)
    px = np.dstack([rgb, rgb, rgb, alpha])[::-1].astype(np.float32)  # Blender's rows start at the bottom
    img.pixels.foreach_set(px.ravel())
    img.filepath_raw = path
    img.file_format = "PNG"
    img.save()
    return img, cols


def cards(name, lines, head_c, human, img, cols, rgb, width=0.016, seed=13):
    """Tapered ribbons along the strands, facing out from the head, as one mesh following head and upper back."""
    rng = np.random.default_rng(seed)
    rig = human.objects.rig
    neck_z = np.array(rig.matrix_world @ rig.data.bones["neck"].head_local)[2]
    verts, faces, uvs, w_head = [], [], [], []
    for line in lines:
        n = len(line)
        col = rng.integers(0, cols)
        wid = width * rng.uniform(0.7, 1.3)
        base = len(verts)
        for i, p in enumerate(line):
            t = i / (n - 1)
            tang = line[min(i + 1, n - 1)] - line[max(i - 1, 0)]
            tang /= np.linalg.norm(tang) + 1e-9
            outw = p - head_c
            outw /= np.linalg.norm(outw) + 1e-9
            side = np.cross(tang, outw)
            side /= np.linalg.norm(side) + 1e-9
            half = wid * (1 - 0.55 * t) / 2
            for sgn, u in ((-1, 0.0), (1, 1.0)):
                verts.append(tuple(p + side * half * sgn))
                uvs.append(((col + 0.05 + 0.9 * u) / cols, 1 - t))
                w_head.append(float(np.clip((p[2] - (neck_z - 0.05)) / 0.12, 0, 1)))
        for i in range(n - 1):
            a = base + 2 * i
            faces.append((a, a + 1, a + 3, a + 2))
    me = bpy.data.meshes.new(name)
    me.from_pydata(verts, [], faces)
    uv = me.uv_layers.new(name="UVMap")
    for f in me.polygons:
        f.use_smooth = True
        for li in f.loop_indices:
            uv.data[li].uv = uvs[me.loops[li].vertex_index]
    obj = bpy.data.objects.new(name, me)
    for col_ in rig.users_collection:
        col_.objects.link(obj)
    obj.parent = rig
    obj.matrix_parent_inverse = rig.matrix_world.inverted()
    mod = obj.modifiers.new("Armature", "ARMATURE")
    mod.object = rig
    gh = obj.vertex_groups.new(name="head")
    gs = obj.vertex_groups.new(name="spine.003")
    for i, w in enumerate(w_head):
        gh.add([i], w, "REPLACE")
        gs.add([i], 1 - w, "REPLACE")
    mat = bpy.data.materials.new("Hair_Cards")
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = img
    mix = nt.nodes.new("ShaderNodeMix")  # the texture's shading times the hair colour
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs[0].default_value = 1.0
    nt.links.new(t.outputs["Color"], mix.inputs[6])
    mix.inputs[7].default_value = (*rgb, 1)
    nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    cut = nt.nodes.new("ShaderNodeMath")
    cut.operation = "GREATER_THAN"
    cut.inputs[1].default_value = 0.3
    nt.links.new(t.outputs["Alpha"], cut.inputs[0])
    nt.links.new(cut.outputs[0], bsdf.inputs["Alpha"])
    bsdf.inputs["Roughness"].default_value = 0.6
    mat.use_backface_culling = False
    me.materials.append(mat)
    return obj


def build(human, length_m, wave, rgb, tex_dir, P_body, N_body):
    lines, head_c = strands(human)
    grown = grow(lines, length_m, wave, head_c, P_body, N_body)
    os.makedirs(tex_dir, exist_ok=True)
    img, cols = strand_texture(os.path.join(tex_dir, "long_hair_strands.png"))
    # bake the colour into the texture: glTF exports image x factor, not a multiply node
    W, H = img.size
    px = np.empty(W * H * 4, np.float32)
    img.pixels.foreach_get(px)
    px = px.reshape(-1, 4)
    lin = px[:, :3] * np.asarray(rgb)[None, :]
    px[:, :3] = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(np.clip(lin, 0, 1), 1 / 2.4) - 0.055)
    img.pixels.foreach_set(px.ravel())
    img.save()
    obj = cards("Hair_Long", grown, head_c, human, img, cols, (1.0, 1.0, 1.0))
    print("LONG_HAIR", len(grown), "strands", f"length {length_m:.2f} m" if length_m else "(style's own length)",
          f"wave {wave:.2f}")
    return obj
