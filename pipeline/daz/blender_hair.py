"""Head hair grown in Blender on an imported Daz Genesis 9 character (hair curves: light, EEVEE-ready).

Used by blender_daz_render.py (--hair hair.json). The scalp is the head's skin above a hairline (set per direction
round the head: forehead, temples, over the ears, nape). Guide strands rise off the scalp, are combed out from the
crown, fall with gravity, keep outside the head and are cut where the style ends (the fringe at the brows, the sides
at mid-ear, the back at the nape: per direction). Every strand follows its nearest guide and gathers into it towards
the tip (spiky locks); a little frizz on top. A thin cap in the hair's colour under the roots hides the bald scalp.
Parented to the head bone. Options (hair.json; defaults by "sex"): strands, guides, length_cm, messy (0..1: spread of
the comb), volume (0..1: lift off the scalp), clump (0..1), frizz_mm, hairline_cm {front, temple, ear, nape: height
above the eyes}, cut_cm {front, side, back: where the strands end, height above the eyes}, flick (share of locks
whose tips turn out and up), wave_mm, shade (colour multiplier), rgb (portrait's hair colour, sRGB 0-255) or colour
"#RRGGBB", seed. Prints HAIR grown ...
"""
import math

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree
from mathutils.kdtree import KDTree

DEFAULTS = {
    "male": {"strands": 32000, "guides": 900, "length_cm": 11, "messy": 0.7, "volume": 0.9, "clump": 0.75, "frizz_mm": 2.5,
             "flick": 0.3, "wave_mm": 2.0, "shade": 0.85,
             "hairline_cm": {"front": 5.6, "temple": 4.6, "ear": 2.0, "nape": -7.5},
             "cut_cm": {"front": 0.6, "side": -1.5, "back": -7.0}},
    "female": {"strands": 30000, "guides": 900, "length_cm": 32, "messy": 0.25, "volume": 0.5, "clump": 0.6, "frizz_mm": 2.0,
               "flick": 0.1, "wave_mm": 3.0, "shade": 0.7,
               "hairline_cm": {"front": 5.8, "temple": 4.8, "ear": 2.2, "nape": -7.5},
               "cut_cm": {"front": 0.0, "side": -14.0, "back": -18.0}},
}


def _interp_az(az, table):
    """Value by direction round the head: az in degrees, 0 = front, 90 = side, 180 = back (|az|)."""
    xs, ys = zip(*table)
    return np.interp(np.abs(az), xs, ys)


def _lin(c):
    c = np.asarray(c, float) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def grow(rig, eye, cfg):
    sex = cfg.get("sex", "female")
    o = dict(DEFAULTS["male" if sex == "male" else "female"])
    for k, v in cfg.items():
        if isinstance(v, dict) and isinstance(o.get(k), dict):
            o[k] = {**o[k], **v}
        else:
            o[k] = v
    rng = np.random.default_rng(int(o.get("seed", 7)))
    sc = bpy.context.scene
    body = next(ob for ob in sc.objects if ob.type == "MESH" and ob.name.startswith("Genesis 9 Mesh"))
    dg = bpy.context.evaluated_depsgraph_get()
    be = body.evaluated_get(dg)
    me = be.to_mesh()
    M = np.array(body.matrix_world)
    V = np.array([v.co for v in me.vertices]) @ M[:3, :3].T + M[:3, 3]
    me.calc_loop_triangles()
    tris = np.array([t.vertices[:] for t in me.loop_triangles])
    tmat = np.array([t.material_index for t in me.loop_triangles])
    head_idx = next((i for i, m in enumerate(body.data.materials) if m and m.name.split(".")[0].split("-")[0] == "Head"), None)
    bvh = BVHTree.FromPolygons([tuple(v) for v in V], [tuple(t) for t in tris])
    E = np.array(eye)
    C = E + np.array([0, 0.035, 0.025])  # the skull's middle, behind the eyes (figure faces -y)

    def az_of(P):
        d = P - C
        return np.degrees(np.arctan2(d[:, 0], -d[:, 1]))

    hl = o["hairline_cm"]
    hair_table = [(0, hl["front"]), (35, (hl["front"] + hl["temple"]) / 2), (65, hl["temple"]), (95, hl["ear"]),
                  (125, hl["ear"] - 1.5), (155, (hl["ear"] + hl["nape"]) / 2), (180, hl["nape"])]
    ct = o["cut_cm"]
    cut_table = [(0, ct["front"]), (40, ct["front"] - 0.5), (80, ct["side"]), (110, ct["side"] - 1.0),
                 (150, (ct["side"] + ct["back"]) / 2), (180, ct["back"])]

    # the scalp: head triangles above the hairline, facing outwards
    cen = V[tris].mean(1)
    nrm = np.cross(V[tris[:, 1]] - V[tris[:, 0]], V[tris[:, 2]] - V[tris[:, 0]])
    area = np.linalg.norm(nrm, axis=1) / 2
    nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
    out = ((cen - C) * nrm).sum(1) > 0
    above = (cen[:, 2] - E[2]) * 100 > _interp_az(az_of(cen), hair_table)
    scalp = np.where((tmat == head_idx if head_idx is not None else True) & out & above & (area > 0))[0]
    if not len(scalp):
        raise SystemExit("HAIR: no scalp found")

    def sample(n):
        f = rng.choice(scalp, n, p=area[scalp] / area[scalp].sum())
        a, b = rng.random(n), rng.random(n)
        sw = a + b > 1
        a[sw], b[sw] = 1 - a[sw], 1 - b[sw]
        P = V[tris[f, 0]] * (1 - a - b)[:, None] + V[tris[f, 1]] * a[:, None] + V[tris[f, 2]] * b[:, None]
        return P, nrm[f]

    K = 16
    W = C + np.array([0, 0.045, 0.09])  # the crown's whorl: the hair is combed out from here

    def grow_guide(r, n):
        t = (r - W) - ((r - W) @ n) * n
        t = t / max(np.linalg.norm(t), 1e-9)
        a = rng.normal(0, o["messy"] * 0.6)  # the comb's direction spread (messy)
        t = t * math.cos(a) + np.cross(n, t) * math.sin(a)
        L = o["length_cm"] / 100 * rng.uniform(0.75, 1.05)
        seg = L / (K * 1.5)  # grown longer than needed, cut at the style's end
        d = n * (0.6 + o["volume"] * 0.8) + t
        d = d / np.linalg.norm(d)
        pts = [r.copy()]
        cut = _interp_az(az_of(r[None]), cut_table)[0] / 100 + E[2] + rng.normal(0, 0.006 * (0.5 + o["messy"]))
        flick = rng.random() < o["flick"]  # this lock's tip turns out and up
        side = np.cross(t, n)
        phase, wl = rng.uniform(0, 2 * math.pi), rng.uniform(0.035, 0.06)
        p = r.copy()
        for i in range(int(K * 1.5)):
            s = i / (K * 1.5)
            d = d + np.array([0, 0, -1.0]) * (0.10 + 0.25 * s) + t * 0.15
            if flick and s > 0.55:  # a gentle turn: the gravity it would have had, plus a little outwards
                out_ = (p - C) / np.linalg.norm(p - C)
                d = d + out_ * 0.12 + np.array([0, 0, 0.30 + 0.25 * s])
            d = d / np.linalg.norm(d)
            p = p + d * seg + side * (o["wave_mm"] / 1000) * (math.sin(phase + i * seg / wl * 2 * math.pi)
                                                             - math.sin(phase + (i - 1) * seg / wl * 2 * math.pi))
            loc, nn, _, _ = bvh.find_nearest(Vector(p))
            if loc is not None:
                off = 0.003 + o["volume"] * 0.010 * (1 - s)
                if (p - np.array(loc)) @ np.array(nn) < off:
                    p = np.array(loc) + np.array(nn) * off
            pts.append(p.copy())
            if p[2] < cut or np.linalg.norm(np.diff(pts, axis=0), axis=1).sum() > L:
                break
        pts = np.array(pts)
        if len(pts) < 3:
            pts = np.array([r, r + n * 0.004, r + n * 0.008])
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        u = np.linspace(0, arc[-1], K + 1)
        return np.c_[[np.interp(u, arc, pts[:, j]) for j in range(3)]].T

    G = int(o["guides"])
    gr, gn = sample(G)
    guides = np.array([grow_guide(r, n) for r, n in zip(gr, gn)])
    kd = KDTree(G)
    for i, r in enumerate(gr):
        kd.insert(Vector(r), i)
    kd.balance()
    N = int(o["strands"])
    sr, sn = sample(N)
    s = np.linspace(0, 1, K + 1)
    strands = np.zeros((N, K + 1, 3))
    for i, r in enumerate(sr):
        _, gi, _ = kd.find(Vector(r))
        g = guides[gi]
        off = r - gr[gi]
        pull = 1 - o["clump"] * s ** 0.8
        frz = rng.normal(0, o["frizz_mm"] / 1000, 3)
        st = g + off[None] * pull[:, None] + frz[None] * s[:, None] ** 1.5
        keep = rng.uniform(0.7, 1.0)  # uneven ends
        arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(st, axis=0), axis=1))]
        u = np.linspace(0, arc[-1] * keep, K + 1)
        strands[i] = np.c_[[np.interp(u, arc, st[:, j]) for j in range(3)]].T

    # the hair object (curves), its material, parented to the head bone
    cv = bpy.data.hair_curves.new("Hair (Blender)")
    cv.add_curves([K + 1] * N)
    cv.attributes["position"].data.foreach_set("vector", strands.reshape(-1).astype(np.float32))
    rad = cv.attributes.get("radius") or cv.attributes.new("radius", "FLOAT", "POINT")
    rad.data.foreach_set("value", np.tile(np.linspace(0.00011, 0.00003, K + 1), N).astype(np.float32))
    ob = bpy.data.objects.new("Hair (Blender)", cv)
    sc.collection.objects.link(ob)
    rgb = _lin(o["rgb"]) if "rgb" in o else _lin([int(o.get("colour", "#2a2a2a").lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)])
    rgb = rgb * o["shade"]  # a hair shader's highlights lift the look: the colour itself darker than the photo's median
    mat = bpy.data.materials.new("Hair (Blender)")
    mat.use_nodes = True
    nt = mat.node_tree
    for nd in list(nt.nodes):
        nt.nodes.remove(nd)
    hb = nt.nodes.new("ShaderNodeBsdfHairPrincipled")
    hb.parametrization = "COLOR"
    hb.inputs["Color"].default_value = (*rgb, 1)
    hb.inputs["Roughness"].default_value = 0.35
    hb.inputs["Radial Roughness"].default_value = 0.45
    outn = nt.nodes.new("ShaderNodeOutputMaterial")
    nt.links.new(hb.outputs[0], outn.inputs["Surface"])
    cv.materials.append(mat)
    # the cap: the scalp a little inside the hairline, 0.7 mm out, in the hair's colour (darker)
    capf = scalp[(cen[scalp, 2] - E[2]) * 100 > _interp_az(az_of(cen[scalp]), hair_table) + 0.6]
    vid = np.unique(tris[capf])
    remap = {v: i for i, v in enumerate(vid)}
    vn = np.zeros((len(V), 3))
    for f in capf:
        vn[tris[f]] += nrm[f]
    vn = vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)
    cm = bpy.data.meshes.new("Hair cap (Blender)")
    cm.from_pydata([tuple(V[v] + vn[v] * 0.0007) for v in vid], [], [tuple(remap[v] for v in tris[f]) for f in capf])
    cap = bpy.data.objects.new("Hair cap (Blender)", cm)
    sc.collection.objects.link(cap)
    capm = bpy.data.materials.new("Hair cap (Blender)")
    capm.use_nodes = True
    bsdf = capm.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*(rgb * 0.55), 1)
    bsdf.inputs["Roughness"].default_value = 0.7
    cm.materials.append(capm)
    for obj in (ob, cap):
        mw = obj.matrix_world.copy()
        obj.parent = rig
        obj.parent_type = "BONE"
        obj.parent_bone = "head"
        obj.matrix_world = mw
    be.to_mesh_clear()
    print("HAIR grown: %d strands from %d guides, %d scalp triangles, colour %s" % (
        N, G, len(scalp), [round(float(x), 3) for x in rgb]), flush=True)
    return ob
