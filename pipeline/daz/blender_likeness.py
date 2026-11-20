"""Face likeness on the imported Daz character in Blender, written back into its Daz face morph.

Used by blender_daz_render.py (--photo). The character as Daz builds it (skin, eyes, lids, lashes) is rendered
straight on with EEVEE (small, fast), MediaPipe finds its landmarks and the portrait's; lined up by scale and position
only, what still differs is the face's shape (eyes opened wider, corners closer, a heavier jaw...). Each landmark's
surface point moves in the picture's plane, a smooth field carries that to the skin near the landmarks (shape key
"Likeness" on the body and on the lashes, brows and tears), a few damped rounds. The body's moves are then added to
the Daz morph (data/.../Morphs/Ours/Faces/<morph>.dsf: same vertex order as Genesis 9's base mesh), so Daz Studio's
character carries the likeness too.
"""
import json
import math
import os
import subprocess
import sys

import bpy
import numpy as np
from mathutils import Vector

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "humgen"))
from hg_landmarks import rbf_moves  # noqa: E402

PY = os.path.join(ROOT, "tools", "pose-env", "Scripts", "python.exe")
INNER_LIPS = {78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191}
# the eyelids' edges, upper and lower, each side (MediaPipe's eye contour)
LIDS = [[246, 161, 160, 159, 158, 157, 173], [7, 163, 144, 145, 153, 154, 155],
        [466, 388, 387, 386, 385, 384, 398], [249, 390, 373, 374, 380, 381, 382]]
FOLLOW = ("Eyelashes", "Eyebrows", "Tear", "Brow", "Lash")  # face meshes that move with the skin


def landmarks(paths, work):
    out = os.path.join(work, "likeness_landmarks.json")
    subprocess.run([PY, "-I", os.path.join(HERE, "landmarks2d.py"), out, *paths], capture_output=True, text=True)
    d = json.load(open(out))
    landmarks.colours = d.get("_colours", {})
    return [None if d.get(p) is None else np.array(d[p]) for p in paths]


def sim2(a, b):
    """Similarity a -> b in 2D (Umeyama): b ~ s R a + t."""
    ma, mb = a.mean(0), b.mean(0)
    A, B = a - ma, b - mb
    U, sv, Vt = np.linalg.svd(B.T @ A)
    dd = np.diag([1, np.sign(np.linalg.det(U @ Vt))])
    R = U @ dd @ Vt
    return (sv * np.diag(dd)).sum() / (A ** 2).sum(), R, mb - ((sv * np.diag(dd)).sum() / (A ** 2).sum()) * (R @ ma)


def run(photo, body, eye, work, morph_path=None, rounds=3, res=(480, 600)):
    sc = bpy.context.scene
    hide = [o for o in sc.objects if o.type == "MESH" and "hair" in o.name.lower() and not any(k in o.name for k in FOLLOW)]
    for o in hide:  # the forehead and brows must be seen
        o.hide_render = True
    cam_d = bpy.data.cameras.new("like_cam")
    cam = bpy.data.objects.new("like_cam", cam_d)
    sc.collection.objects.link(cam)
    old = (sc.camera, sc.render.resolution_x, sc.render.resolution_y, sc.render.filepath, sc.render.film_transparent)
    sc.camera = cam
    cam_d.lens = 85
    tgt = eye + Vector((0, 0, -0.04))
    cam.location = tgt + Vector((0, -0.75, 0))
    cam.rotation_euler = (tgt - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.render.resolution_x, sc.render.resolution_y = res
    sc.render.film_transparent = False
    try:
        samples = sc.eevee.taa_render_samples
        sc.eevee.taa_render_samples = 16
    except Exception:
        samples = None
    W, H = res
    frame = [cam_d.view_frame(scene=sc)[i] for i in range(4)]  # camera space corners at depth 1 (tr, br, bl, tl)
    xs, ys = [v.x for v in frame], [v.y for v in frame]
    fx0, fx1, fy0, fy1, fz = min(xs), max(xs), min(ys), max(ys), frame[0].z

    # every mesh that follows the face, with a shape key to hold the moves
    movers = [body] + [o for o in sc.objects if o.type == "MESH" and any(k in o.name for k in FOLLOW)]
    for o in movers:
        if not o.data.shape_keys:
            o.shape_key_add(name="Basis")
        k = o.data.shape_keys.key_blocks.get("Likeness") or o.shape_key_add(name="Likeness")
        k.value = 1.0
    base_world = {o.name: np.array([o.matrix_world @ v.co for v in o.data.vertices]) for o in movers}
    total = {o.name: np.zeros_like(base_world[o.name]) for o in movers}

    q = landmarks([photo], work)[0]
    if q is None:
        print("LIKENESS skipped: no face on", photo)
        return None
    best = (np.inf, None)
    for rnd in range(rounds + 1):
        img = os.path.join(work, f"likeness_{rnd}.png")
        sc.render.filepath = img
        bpy.ops.render.render(write_still=True)
        r = landmarks([img], work)[0]
        if r is None:
            print("LIKENESS round", rnd, ": no face found on the render")
            break
        ok = ~np.isin(np.arange(468), list(INNER_LIPS))
        Q, Rp = q[:468][ok], r[:468][ok]
        s, R2, t = sim2(Q, Rp)
        T = (s * (R2 @ Q.T)).T + t  # the portrait's landmarks on the render, same size and place
        res2 = T - Rp
        # detection noise out, a region's shift kept: each landmark with its nearest neighbours of the same part
        # (an upper lid's points move against the lower lid's: never averaged together)
        part = np.zeros(468, int)
        for k_, ids_ in enumerate(LIDS):
            part[ids_] = k_ + 1
        pk = part[np.arange(468)[ok]]
        dist = np.linalg.norm(Rp[:, None] - Rp[None], axis=2) + 1e6 * (pk[:, None] != pk[None])
        nn = np.argsort(dist, axis=1)[:, :5]
        res2 = np.median(res2[nn], axis=1)
        # trace each render landmark onto the body's surface
        dg = bpy.context.evaluated_depsgraph_get()
        cam_m = cam.matrix_world
        be = body.evaluated_get(dg)
        bw, bi = body.matrix_world, body.matrix_world.inverted()
        fwd = (cam_m.to_3x3() @ Vector((0, 0, -1))).normalized()
        hits, moves, depths = [], [], []
        for (u, v), (du, dv) in zip(Rp, res2):
            d_cam = Vector((fx0 + (u / W) * (fx1 - fx0), fy1 - (v / H) * (fy1 - fy0), fz))
            ray = (cam_m.to_3x3() @ d_cam).normalized()
            # the body only (lashes in front of the lid's edge would hide it)
            hit, loc, nrm, idx = be.ray_cast(bi @ cam_m.translation, (bi.to_3x3() @ ray).normalized())
            if not hit:
                continue
            loc = bw @ loc
            depth = (loc - cam_m.translation).dot(fwd)
            depths.append(depth)
            px = depth * (fx1 - fx0) / abs(fz) / W  # metres per pixel at that depth
            right, up = cam_m.to_3x3() @ Vector((1, 0, 0)), cam_m.to_3x3() @ Vector((0, 1, 0))
            hits.append(np.array(loc))
            moves.append(np.array(right * (du * px) + up * (-dv * px)))
        hits, moves, depths = np.array(hits), np.array(moves), np.array(depths)
        front = depths < np.median(depths) + 0.03  # a ray through the eye's opening hits far behind: dropped
        hits, moves = hits[front], moves[front]
        mm = np.linalg.norm(moves, axis=1) * 1000
        iod = np.linalg.norm(r[33] - r[263])
        open_ = (np.linalg.norm(r[159] - r[145]) + np.linalg.norm(r[386] - r[374])) / 2 / iod
        open_p = (np.linalg.norm(q[159] - q[145]) + np.linalg.norm(q[386] - q[374])) / 2 / np.linalg.norm(q[33] - q[263])
        print("LIKENESS round %d: %d landmarks on the skin, off by median %.2f mm, largest %.2f mm; eye opening %.3f "
              "(portrait %.3f)" % (rnd, len(hits), np.median(mm), mm.max(), open_, open_p), flush=True)
        if np.median(mm) < best[0]:
            best = (float(np.median(mm)), {k: v.copy() for k, v in total.items()})
        elif np.median(mm) > 1.15 * best[0]:
            print("LIKENESS back to the best round (median %.2f mm)" % best[0])
            break
        if rnd == rounds or np.median(mm) < 0.3:
            break
        n = np.linalg.norm(moves, axis=1, keepdims=True)
        moves = moves * np.minimum(1, 0.004 / np.maximum(n, 1e-9)) * 0.85  # at most 4 mm a round, damped
        for o in movers:
            P = base_world[o.name] + total[o.name]
            dn = np.min(np.linalg.norm(P[:, None] - hits[None], axis=2), axis=1) if len(P) < 9000 else None
            if dn is None:  # big mesh: only points near the face get the field
                c = hits.mean(0)
                near = np.where(np.linalg.norm(P - c, axis=1) < 0.16)[0]
                dn = np.full(len(P), np.inf)
                for a in range(0, len(near), 3000):
                    ch = near[a:a + 3000]
                    dn[ch] = np.min(np.linalg.norm(P[ch][:, None] - hits[None], axis=2), axis=1)
            wgt = np.clip(1 - (dn - 0.008) / 0.025, 0, 1)  # full on the face, gone 3.3 cm beyond the landmarks
            idx = np.where(wgt > 0)[0]
            if len(idx):
                total[o.name][idx] += rbf_moves(hits, moves, P[idx]) * wgt[idx, None]
            apply(o, base_world[o.name], total[o.name])
    total = best[1] if best[1] is not None else total
    for o in movers:
        apply(o, base_world[o.name], total[o.name])
    # the eyelids: where each lid's edge sits on the iris, as in the portrait (a stare vs a heavy lid)
    total = lids(q, movers, base_world, total, work)
    for o in hide:
        o.hide_render = False
    sc.camera, sc.render.resolution_x, sc.render.resolution_y, sc.render.filepath, sc.render.film_transparent = old
    if samples is not None:
        sc.eevee.taa_render_samples = samples
    bpy.data.objects.remove(cam)
    moved = total[body.name]
    print("LIKENESS body moved median %.2f mm (moving points), largest %.2f mm" % (
        np.median(np.linalg.norm(moved[np.linalg.norm(moved, axis=1) > 1e-5], axis=1)) * 1000 if (np.linalg.norm(moved, axis=1) > 1e-5).any() else 0,
        np.linalg.norm(moved, axis=1).max() * 1000))
    if morph_path:
        write_morph(morph_path, body, moved)
    return moved


LID_BONES = {"r_up": "r_eyelidupper", "l_up": "l_eyelidupper", "r_lo": "r_eyelidlower", "l_lo": "l_eyelidlower"}


def lid_metrics(p):
    """Each lid's edge against its iris, in iris radii (picture y down): upper = how far the upper lid's edge is
    above the iris centre, lower = how far the lower lid's edge is below it. MediaPipe: 468/473 iris centres of the
    subject's right/left eye, 469-472/474-477 their outlines, 159/145 and 386/374 the lids' middles."""
    out = {}
    for side, c, ring, up, lo in (("r", 468, range(469, 473), 159, 145), ("l", 473, range(474, 478), 386, 374)):
        r = np.mean([np.linalg.norm(p[i] - p[c]) for i in ring])
        out[side + "_up"] = (p[c][1] - p[up][1]) / r
        out[side + "_lo"] = (p[lo][1] - p[c][1]) / r
    return out


def lids(q, movers, base_world, total, work, renders=4):
    """Rotate each eyelid's skin about its eye (as Genesis 9's own lid bones do, with their skin weights): the angle
    per lid solved so the lid edge sits on the iris as in the portrait. Baked into the shape (and the Daz morph)."""
    sc = bpy.context.scene
    rig = next((o for o in sc.objects if o.type == "ARMATURE"), None)
    if rig is None or len(q) < 478:
        return total
    want = lid_metrics(q)
    piv = {}
    for k, bn in LID_BONES.items():
        b = rig.pose.bones.get(bn) or rig.pose.bones.get(bn.replace("eyelid", "eye"))
        e = rig.pose.bones.get(k[0] + "_eye")
        if b is None or e is None:
            return total
        piv[k] = np.array(rig.matrix_world @ e.head)  # the eyeball's centre: the lid slides over it
    weights = {}
    for o in movers:
        for k, bn in LID_BONES.items():
            vg = o.vertex_groups.get(bn)
            w = np.zeros(len(o.data.vertices))
            if vg is not None:
                gi = vg.index
                for v in o.data.vertices:
                    for g in v.groups:
                        if g.group == gi:
                            w[v.index] = g.weight
            weights[(o.name, k)] = w

    def delta(o, ang):  # ang: degrees per lid, + closes (upper lid down, lower lid up)
        P = base_world[o.name] + total[o.name]
        d = np.zeros_like(P)
        for k, a in ang.items():
            w = weights[(o.name, k)]
            idx = np.where(w > 1e-3)[0]
            if not len(idx) or abs(a) < 1e-6:
                continue
            t = np.radians(a * (1 if k.endswith("up") else -1)) * w[idx]  # about +x: front points go down for t > 0
            rel = P[idx] - piv[k]
            y, z = rel[:, 1], rel[:, 2]
            d[idx, 1] = y * np.cos(t) - z * np.sin(t) - y
            d[idx, 2] = y * np.sin(t) + z * np.cos(t) - z
        return d

    def measure(ang, tag):
        for o in movers:
            apply(o, base_world[o.name], total[o.name] + delta(o, ang))
        img = os.path.join(work, f"lids_{tag}.png")
        sc.render.filepath = img
        bpy.ops.render.render(write_still=True)
        r = landmarks([img], work)[0]
        return None if r is None or len(r) < 478 else lid_metrics(r)

    keys = list(LID_BONES)
    a0 = {k: 0.0 for k in keys}
    m0 = measure(a0, 0)
    if m0 is None:
        return total
    print("LIDS portrait %s | render %s (iris radii)" % ({k: round(v, 2) for k, v in want.items()},
                                                         {k: round(v, 2) for k, v in m0.items()}), flush=True)
    a1 = {k: (4.0 if m0[k] > want[k] else -4.0) for k in keys}
    m1 = measure(a1, 1)
    hist = [(a0, m0), (a1, m1)]
    for it in range(2, renders):
        (ap, mp_), (ac, mc) = hist[-2], hist[-1]
        if mc is None or mp_ is None:
            break
        an = {}
        for k in keys:
            slope = (mc[k] - mp_[k]) / (ac[k] - ap[k]) if abs(ac[k] - ap[k]) > 1e-6 else 0
            an[k] = float(np.clip(ac[k] + (want[k] - mc[k]) / slope, -6, 18)) if slope < -1e-4 else ac[k]
        hist.append((an, measure(an, it)))
    best_a, best_e = a0, np.inf
    for a, m in hist:
        if m is not None:
            e = sum(abs(m[k] - want[k]) for k in keys)
            if e < best_e:
                best_a, best_e = a, e
    for o in movers:
        total[o.name] = total[o.name] + delta(o, best_a)
        apply(o, base_world[o.name], total[o.name])
    print("LIDS angles (deg, + closes) %s; render %s" % ({k: round(v, 1) for k, v in best_a.items()},
                                                          {k: round(v, 2) for k, v in dict(hist[[h[0] for h in hist].index(best_a)][1]).items()}), flush=True)
    return total


def apply(o, base_world, total):
    inv = o.matrix_world.inverted().to_3x3()
    key = o.data.shape_keys.key_blocks["Likeness"]
    basis = o.data.shape_keys.key_blocks["Basis"]
    for i, (b, t) in enumerate(zip(basis.data, total)):
        key.data[i].co = b.co + inv @ Vector(t)


def write_morph(path, body, moved_world):
    """Add the body's moves to the Daz morph (Daz frame: cm, y up, facing +z)."""
    d = json.load(open(path))
    mod = d["modifier_library"][0]
    vals = mod["morph"]["deltas"]["values"]
    n_base = len(body.data.vertices)
    cur = {int(v[0]): np.array(v[1:4], float) for v in vals}
    dm = np.c_[moved_world[:, 0], moved_world[:, 2], -moved_world[:, 1]] * 100.0
    for i in np.where(np.linalg.norm(dm, axis=1) > 1e-4)[0]:
        cur[int(i)] = cur.get(int(i), np.zeros(3)) + dm[i]
    mod["morph"]["deltas"] = {"count": len(cur), "values": [[i, *[round(float(x), 5) for x in v]] for i, v in sorted(cur.items())]}
    mod["morph"]["vertex_count"] = mod["morph"].get("vertex_count", n_base)
    json.dump(d, open(path, "w"))
    print("LIKENESS written into the Daz morph", path, len(cur), "points")
