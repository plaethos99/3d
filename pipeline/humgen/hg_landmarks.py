"""Which HG body vertex each MediaPipe face landmark is (for landmark-driven face capture).

hg_landmark_vertices(human, P, polys, cache) -> (478,) vertex indices (-1 where not found)
Renders the HG face straight on (orthographic, hair and clothes hidden), runs MediaPipe's face landmarker on it
(pose-env python), and traces each landmark back onto the mesh along the camera's view; the nearest vertex of the
hit face is the landmark's. Every HG human of a gender shares the mesh, so the result is cached (cache path).
"""
import json
import os
import subprocess

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PY = os.path.join(ROOT, "tools", "pose-env", "Scripts", "python.exe")
MODEL = os.path.join(ROOT, "tools", "face-model", "face_landmarker.task")
DETECT = r'''
import sys, json, cv2, mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
lm = FaceLandmarker.create_from_options(FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=sys.argv[1]),
                                        running_mode=RunningMode.IMAGE, num_faces=1))
im = cv2.imread(sys.argv[2]); h, w = im.shape[:2]
r = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
print("LANDMARKS", json.dumps([[q.x * w, q.y * h] for q in r.face_landmarks[0]] if r.face_landmarks else []))
'''


def hg_landmark_vertices(human, P, polys, cache, work_dir):
    if cache and os.path.exists(cache):
        d = json.load(open(cache))
        if d.get("vertices") == len(P):
            return np.array(d["landmarks"])
    sc = bpy.context.scene
    body = human.objects.body
    rig = human.objects.rig
    hidden = []
    for o in bpy.data.objects:  # just the body, eyes and teeth
        if o.type == "MESH" and o not in (body, human.objects.eyes) and not o.hide_render:
            o.hide_render = True
            hidden.append(o)
    ps_off = [m for m in body.modifiers if m.type == "PARTICLE_SYSTEM" and m.show_render]
    for m in ps_off:
        m.show_render = False
    eyes = [np.array(rig.matrix_world @ rig.data.bones[n].head_local) for n in ("eyeball.L", "eyeball.R")]
    c = (eyes[0] + eyes[1]) / 2 + np.array([0, 0, -0.03])
    scale, res = 0.30, 1024
    cam = bpy.data.objects.new("lm_cam", bpy.data.cameras.new("lm_cam"))
    sc.collection.objects.link(cam)
    cam.data.type = "ORTHO"
    cam.data.ortho_scale = scale
    cam.location = Vector((c[0], c[1] - 1.0, c[2]))
    cam.rotation_euler = (Vector(c) - cam.location).to_track_quat("-Z", "Y").to_euler()
    light = bpy.data.objects.new("lm_light", bpy.data.lights.new("lm_light", "SUN"))
    sc.collection.objects.link(light)
    light.rotation_euler = (1.3, 0, 0.2)
    light.data.energy = 3
    old = (sc.camera, sc.render.resolution_x, sc.render.resolution_y, sc.render.filepath, sc.render.engine)
    sc.camera = cam
    sc.render.engine = "BLENDER_EEVEE"
    sc.render.resolution_x = sc.render.resolution_y = res
    img = os.path.join(work_dir, "hg_neutral_front.png")
    sc.render.filepath = img
    bpy.ops.render.render(write_still=True)
    sc.camera, sc.render.resolution_x, sc.render.resolution_y, sc.render.filepath, sc.render.engine = old
    bpy.data.objects.remove(cam)
    bpy.data.objects.remove(light)
    for o in hidden:
        o.hide_render = False
    for m in ps_off:
        m.show_render = True
    out = subprocess.run([PY, "-I", "-c", DETECT, MODEL, img], capture_output=True, text=True)
    line = next((ln for ln in out.stdout.splitlines() if ln.startswith("LANDMARKS")), None)
    pts2 = json.loads(line.split(" ", 1)[1]) if line else []
    if not pts2:
        raise RuntimeError("MediaPipe found no face on the HG render: " + out.stderr[-300:])
    tree = BVHTree.FromPolygons([tuple(p) for p in P], polys)
    found = []
    for x, y in pts2:
        wx = c[0] + (x / res - 0.5) * scale
        wz = c[2] - (y / res - 0.5) * scale
        hit = tree.ray_cast(Vector((wx, c[1] - 1.0, wz)), Vector((0, 1, 0)))
        if hit[0] is None:
            found.append(-1)
            continue
        poly = polys[hit[2]]
        pos = np.array(hit[0])
        found.append(int(min(poly, key=lambda vi: np.linalg.norm(P[vi] - pos))))
    if cache:
        json.dump({"vertices": len(P), "landmarks": found}, open(cache, "w"))
    return np.array(found)


def surface_landmarks(P, fs, work_dir, tag="mesh", colour=(0.8, 0.62, 0.52)):
    """MediaPipe's 478 landmarks on any head mesh facing -y (z up): a straight-on orthographic render, the
    landmarks found on it, each traced back onto the mesh. Returns (vertex index per landmark, -1 if missed;
    the surface point hit, (478, 3)). The camera frames the head's top 0.35 of its height plus the face."""
    sc = bpy.context.scene
    me = bpy.data.meshes.new("lm_" + tag)
    me.from_pydata([tuple(p) for p in P], [], fs)
    me.update()
    ob = bpy.data.objects.new("lm_" + tag, me)
    sc.collection.objects.link(ob)
    mat = bpy.data.materials.new("lm_skin")
    mat.diffuse_color = (*colour, 1)
    if mat.node_tree and mat.node_tree.nodes.get("Principled BSDF"):
        mat.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (*colour, 1)
    ob.data.materials.append(mat)
    front = P[P[:, 1] < np.percentile(P[:, 1], 30)]  # the front of the head: the face
    c = np.array([np.median(front[:, 0]), P[:, 1].min(), np.median(front[:, 2])])
    width = float(np.percentile(front[:, 0], 98) - np.percentile(front[:, 0], 2))
    res = 1024
    pts2 = []
    for zoom in (1.6, 2.3, 3.2):  # wider framing until MediaPipe sees a whole face
        scale = zoom * width
        cam = bpy.data.objects.new("lm_cam", bpy.data.cameras.new("lm_cam"))
        sc.collection.objects.link(cam)
        cam.data.type = "ORTHO"
        cam.data.ortho_scale = scale
        cam.data.clip_end = 100
        cam.location = Vector((c[0], c[1] - scale * 3, c[2]))
        cam.rotation_euler = (Vector(c) - cam.location).to_track_quat("-Z", "Y").to_euler()
        light = bpy.data.objects.new("lm_light", bpy.data.lights.new("lm_light", "SUN"))
        sc.collection.objects.link(light)
        light.rotation_euler = (1.3, 0, 0.2)
        light.data.energy = 3
        hidden = [o for o in sc.objects if o not in (ob, cam, light) and not o.hide_render]
        for o in hidden:
            o.hide_render = True
        old = (sc.camera, sc.render.resolution_x, sc.render.resolution_y, sc.render.filepath)
        sc.camera = cam
        sc.render.resolution_x = sc.render.resolution_y = res
        img = os.path.join(work_dir, f"{tag}_front_landmarks.png")
        sc.render.filepath = img
        bpy.ops.render.render(write_still=True)
        sc.camera, sc.render.resolution_x, sc.render.resolution_y, sc.render.filepath = old
        for o in hidden:
            o.hide_render = False
        for o in (cam, light):
            bpy.data.objects.remove(o)
        out = subprocess.run([PY, "-I", "-c", DETECT, MODEL, img], capture_output=True, text=True)
        line = next((ln for ln in out.stdout.splitlines() if ln.startswith("LANDMARKS")), None)
        pts2 = json.loads(line.split(" ", 1)[1]) if line else []
        if pts2:
            break
    bpy.data.objects.remove(ob)
    if not pts2:
        raise RuntimeError("MediaPipe found no face on the render " + img + " " + out.stderr[-300:])
    tree = BVHTree.FromPolygons([tuple(p) for p in P], fs)
    ids, hits = [], []
    for x, y in pts2:
        wx = c[0] + (x / res - 0.5) * scale
        wz = c[2] - (y / res - 0.5) * scale
        hit = tree.ray_cast(Vector((wx, c[1] - scale * 3, wz)), Vector((0, 1, 0)))
        if hit[0] is None:
            ids.append(-1)
            hits.append((np.nan, np.nan, np.nan))
            continue
        poly = fs[hit[2]]
        ids.append(int(min(poly, key=lambda vi: np.linalg.norm(P[vi] - np.array(hit[0])))))
        hits.append(tuple(hit[0]))
    return np.array(ids), np.array(hits)


def similarity(src, dst, w=None):
    """Umeyama: s, R, t with dst ~ s R src + t (weighted)."""
    w = np.ones(len(src)) if w is None else np.asarray(w, float)
    w = w / w.sum()
    ms, md = (w[:, None] * src).sum(0), (w[:, None] * dst).sum(0)
    xs, xd = src - ms, dst - md
    cov = (w[:, None] * xd).T @ xs
    U, S, Vt = np.linalg.svd(cov)
    dfix = np.eye(3)
    dfix[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ dfix @ Vt
    var = (w * (xs ** 2).sum(1)).sum()
    s = float(np.trace(np.diag(S) @ dfix) / var)
    return s, R, md - s * R @ ms


def rbf_moves(centres, displacements, points, smooth=1e-5):
    """Biharmonic (phi = r) interpolation of displacements at centres to points, with a linear part."""
    n = len(centres)
    K = np.linalg.norm(centres[:, None] - centres[None], axis=2)
    Pm = np.c_[np.ones(n), centres]
    A = np.zeros((n + 4, n + 4))
    A[:n, :n] = K + smooth * np.eye(n)
    A[:n, n:] = Pm
    A[n:, :n] = Pm.T
    rhs = np.zeros((n + 4, 3))
    rhs[:n] = displacements
    sol = np.linalg.solve(A, rhs)
    out = np.zeros((len(points), 3))
    for a in range(0, len(points), 4000):
        chunk = points[a:a + 4000]
        Kp = np.linalg.norm(chunk[:, None] - centres[None], axis=2)
        out[a:a + 4000] = Kp @ sol[:n] + np.c_[np.ones(len(chunk)), chunk] @ sol[n:]
    return out
