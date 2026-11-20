"""FLAME fitted straight to a person's photos: one head shape seen by every photo (our own FaceBuilder-like fit).

Usage: tools/pose-env/Scripts/python.exe -I pipeline/flame/flame_photofit.py <out_dir> <image> [<image> ...]
       [--shapes 50] [--expr 30] [--shape-reg 3.0] [--expr-reg 1.0] [--iters 40] [--tex 2048] [--no-tex]
Images: the same person from several sides (face_from_image.py's stills: front, left45, left75, right45, right75).
1. MediaPipe's 468 face points in every photo.
2. Unknowns: FLAME's first --shapes identity values (one head for all photos), per photo: its --expr expression
   values (a smile in one photo doesn't bend the head), the head's turn and place, and the lens (focal length).
   Solved together (damped Gauss-Newton) so FLAME's matching points land on the photo's points in every view;
   identity and expression values are kept near real faces (ridge, in standard deviations). A point counts in a
   view in proportion to how squarely that view sees it (the far side of a turned face is guessed by MediaPipe).
   Defaults tuned on the schoolteacher against her photo (render vs photo, shape distance): 100 shapes 0.044,
   50 shapes with these regs 0.037 (FaceBuilder 0.034); fewer shapes or a down-weighted outline did worse.
3. The neutral head (no expression) -> <out>/heads.npz (for flame_head.py), with a texture layout (around the
   head, like a cylinder: the face and sides undistorted) and <out>/texture.png baked from the photos: each texel
   from the photo that sees it most squarely (alpha 0 where none does).
Writes fit.json (per view: error in pixels, turn, focal). Prints FLAME_PHOTOFIT_OK.
FLAME 2023 Open: CC-BY-4.0, MPI-IS.
"""
import json
import os
import subprocess
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from flame_model import load  # noqa: E402

args = sys.argv[1:]


def opt(name, default):
    return args[args.index(name) + 1] if name in args else default


flags = {"--shapes", "--expr", "--shape-reg", "--expr-reg", "--iters", "--tex", "--outline-turned"}
no_tex = "--no-tex" in args
args = [a for a in args if a != "--no-tex"]
pos = [a for i, a in enumerate(args) if a not in flags and (i == 0 or args[i - 1] not in flags)]
out, images = os.path.abspath(pos[0]), [os.path.abspath(p) for p in pos[1:]]
os.makedirs(out, exist_ok=True)
KS, KE = int(opt("--shapes", "50")), int(opt("--expr", "30"))
reg_s, reg_e = float(opt("--shape-reg", "3.0")), float(opt("--expr-reg", "1.0"))
iters, tex_n = int(opt("--iters", "40")), int(opt("--tex", "2048"))
# the face outline (jaw, cheeks, forehead edge): in a turned photo MediaPipe puts it on the silhouette, which is
# not a fixed point of the face: those points count this much in photos turned more than 20 degrees
outline_turned = float(opt("--outline-turned", "1.0"))
OUTLINE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176,
           149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

m = load(os.path.join(ROOT, "tools", "flame", "flame2023_Open.pkl"))
T, SD = m["v_template"], m["shapedirs"]
SHAPE, EXPR = SD[:, :, :KS], SD[:, :, m["n_shape"]:m["n_shape"] + KE]
F = m["faces"]
lm = np.array(json.load(open(os.path.join(ROOT, "tools", "flame", "flame_landmarks.json")))["landmarks"])[:468]

# 1. landmarks in every photo
DETECT = r'''
import sys, json, cv2, mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode
lmk = FaceLandmarker.create_from_options(FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=sys.argv[1]),
                                         running_mode=RunningMode.IMAGE, num_faces=1))
res = []
for p in sys.argv[2:]:
    im = cv2.imread(p); h, w = im.shape[:2]
    r = lmk.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
    res.append([[q.x * w, q.y * h] for q in r.face_landmarks[0]][:468] if r.face_landmarks else [])
print("LANDMARKS", json.dumps(res))
'''
r = subprocess.run([sys.executable, "-I", "-c", DETECT, os.path.join(ROOT, "tools", "face-model", "face_landmarker.task"),
                    *images], capture_output=True, text=True)
pts = json.loads(next(ln for ln in r.stdout.splitlines() if ln.startswith("LANDMARKS")).split(" ", 1)[1])
views = [(p, np.array(q)) for p, q in zip(images, pts) if len(q)]
print("PHOTOS with a face:", len(views), "of", len(images))
ok = lm >= 0
idx = np.where(ok)[0]
vi = lm[idx]
A0, AS, AE = T[vi], SHAPE[vi], EXPR[vi]  # (n,3), (n,3,KS), (n,3,KE)
# normals of FLAME's mean head at those points (for how squarely each view sees them)
vn = np.zeros_like(T)
fn = np.cross(T[F[:, 1]] - T[F[:, 0]], T[F[:, 2]] - T[F[:, 0]])
for k in range(3):
    np.add.at(vn, F[:, k], fn)
vn /= np.linalg.norm(vn, axis=1)[:, None] + 1e-12
N0 = vn[vi]


def rodrigues(w):
    th = np.linalg.norm(w)
    if th < 1e-12:
        return np.eye(3)
    k = w / th
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K


# initial pose per photo: solvePnP of FLAME's mean head (its own frame; the solver finds the turn)
cams = []
for path, q in views:
    h, w = cv2.imread(path).shape[:2]
    f0 = 1.6 * max(w, h)
    K = np.array([[f0, 0, w / 2], [0, f0, h / 2], [0, 0, 1]])
    obj, img_pts = A0.astype(np.float64), q[idx].astype(np.float64)
    # start from "facing the camera" (180 deg about x: y up -> down, +z towards the camera), 0.6 m away
    rv0 = np.array([np.pi, 0.0, 0.0])
    tv0 = np.array([0.0, 0.0, 0.6])
    okp, rv, tv = cv2.solvePnP(obj, img_pts, K, None, rv0.reshape(3, 1), tv0.reshape(3, 1), True,
                               flags=cv2.SOLVEPNP_ITERATIVE)
    cams.append({"w": w, "h": h, "rv": rv.ravel(), "t": tv.ravel(), "logf": np.log(f0)})
nv = len(views)
# parameters: shape (KS) | per view: rot(3) trans(3) logf(1) expr(KE)
P_VIEW = 7 + KE


def pack(b, cams_, exprs):
    x = [b]
    for c, e in zip(cams_, exprs):
        x += [c["rv"], c["t"], [c["logf"]], e]
    return np.concatenate([np.ravel(a) for a in x])


def unpack(x):
    b = x[:KS]
    out_ = []
    for v in range(nv):
        o = KS + v * P_VIEW
        out_.append((x[o:o + 3], x[o + 3:o + 6], x[o + 6], x[o + 7:o + 7 + KE]))
    return b, out_


def project(b, rv, t, logf, e, v):
    X = A0 + AS @ b + AE @ e
    R = rodrigues(rv)
    Xc = X @ R.T + t
    f = np.exp(logf)
    uv = f * Xc[:, :2] / Xc[:, 2:3] + np.array([cams[v]["w"] / 2, cams[v]["h"] / 2])
    facing = np.clip(-(N0 @ R.T) @ np.array([0, 0, 1.0]), 0, 1)  # how squarely the camera sees each point
    return uv, facing


def residuals(x, weights):
    b, vs = unpack(x)
    res = []
    for v, (rv, t, logf, e) in enumerate(vs):
        uv, _ = project(b, rv, t, logf, e, v)
        scale = 1.0 / max(cams[v]["w"], cams[v]["h"]) * 1000  # pixels in thousandths of the image
        res.append(((uv - views[v][1][idx]) * scale * np.sqrt(weights[v])[:, None]).ravel())
    res.append(np.sqrt(reg_s) * b)
    for rv, t, logf, e in vs:
        res.append(np.sqrt(reg_e) * e)
    return np.concatenate(res)


x = pack(np.zeros(KS), cams, [np.zeros(KE)] * nv)
lam = 1e-2
for it in range(iters):
    b, vs = unpack(x)
    weights = []
    for v, vv in enumerate(vs):
        w_ = np.clip(project(b, *vv, v)[1], 0.05, 1.0) ** 2
        turn = np.degrees(np.arccos(np.clip(-(rodrigues(vv[0]) @ np.array([0, 0, 1.0]))[2], -1, 1)))
        if turn > 20:
            w_ = w_ * np.where(np.isin(idx, OUTLINE), outline_turned, 1.0)
        weights.append(w_)
    r0 = residuals(x, weights)
    J = np.empty((len(r0), len(x)))
    eps = 1e-5
    for j in range(len(x)):
        xp = x.copy()
        xp[j] += eps
        J[:, j] = (residuals(xp, weights) - r0) / eps
    g, H = J.T @ r0, J.T @ J
    while True:
        dx = -np.linalg.solve(H + lam * np.diag(np.diag(H) + 1e-9), g)
        r1 = residuals(x + dx, weights)
        if r1 @ r1 < r0 @ r0:
            x, lam = x + dx, max(lam / 3, 1e-7)
            break
        lam *= 4
        if lam > 1e8:
            break
    if it % 5 == 0 or it == iters - 1:
        print(f"iter {it}: cost {r0 @ r0:.1f} -> {r1 @ r1:.1f}", flush=True)

b, vs = unpack(x)
report = {"views": []}
for v, (rv, t, logf, e) in enumerate(vs):
    uv, facing = project(b, rv, t, logf, e, v)
    err = np.linalg.norm(uv - views[v][1][idx], axis=1)
    seen = facing > 0.3
    R = rodrigues(rv)
    yaw = float(np.degrees(np.arctan2(R[0, 2], R[2, 2])))
    report["views"].append({"image": os.path.basename(views[v][0]), "error_px_median": round(float(np.median(err[seen])), 2),
                            "error_px_p90": round(float(np.percentile(err[seen], 90)), 2), "yaw_deg": round(yaw, 1),
                            "focal_px": round(float(np.exp(logf)), 1), "expression_rms": round(float(np.sqrt((e ** 2).mean())), 2)})
    print("VIEW", report["views"][-1], flush=True)
report["shape_rms"] = round(float(np.sqrt((b ** 2).mean())), 3)
report["shape_values"] = [round(float(v), 3) for v in b]
print("SHAPE rms (standard deviations):", report["shape_rms"])

# 3. the neutral head, its texture layout, and its texture from the photos
V = T + SHAPE @ b
c = V.mean(0)
ang = np.arctan2(V[:, 0] - c[0], V[:, 2] - c[2])  # 0 = straight ahead (+z)
UV = np.c_[ang / (2 * np.pi) + 0.5, (V[:, 1] - V[:, 1].min()) / np.ptp(V[:, 1])]
np.savez(os.path.join(out, "heads.npz"), faces=F, names=np.array(["photofit"]), verts=V[None], uv=UV)
vnf = np.zeros_like(V)
fnf = np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]])
for k in range(3):
    np.add.at(vnf, F[:, k], fnf)
vnf /= np.linalg.norm(vnf, axis=1)[:, None] + 1e-12
if no_tex:
    json.dump(report, open(os.path.join(out, "fit.json"), "w"), indent=1)
    print("FLAME_PHOTOFIT_OK", out, "(no texture)")
    sys.exit(0)
# rasterize the layout: per texel its 3D point and normal (faces not crossing the back seam)
pos = np.zeros((tex_n, tex_n, 3), np.float32)
nrm = np.zeros((tex_n, tex_n, 3), np.float32)
cov = np.zeros((tex_n, tex_n), bool)
rbuf = np.full((tex_n, tex_n), -1.0, np.float32)  # the outermost surface wins (lips over the mouth's inside)
for f in F:
    uvf = UV[f] * (tex_n - 1)
    if np.ptp(uvf[:, 0]) > tex_n / 2:
        continue
    x0, y0 = np.floor(uvf.min(0)).astype(int)
    x1, y1 = np.ceil(uvf.max(0)).astype(int)
    xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
    pxy = np.c_[xs.ravel(), ys.ravel()].astype(float)
    a, b_, c_ = uvf
    den = (b_[1] - c_[1]) * (a[0] - c_[0]) + (c_[0] - b_[0]) * (a[1] - c_[1])
    if abs(den) < 1e-9:
        continue
    l1 = ((b_[1] - c_[1]) * (pxy[:, 0] - c_[0]) + (c_[0] - b_[0]) * (pxy[:, 1] - c_[1])) / den
    l2 = ((c_[1] - a[1]) * (pxy[:, 0] - c_[0]) + (a[0] - c_[0]) * (pxy[:, 1] - c_[1])) / den
    l3 = 1 - l1 - l2
    inside = (l1 >= -0.01) & (l2 >= -0.01) & (l3 >= -0.01)
    if not inside.any():
        continue
    L = np.c_[l1, l2, l3][inside]
    px = pxy[inside].astype(int)
    P_ = L @ V[f]
    rad_ = np.hypot(P_[:, 0] - c[0], P_[:, 2] - c[2])
    out_ = rad_ > rbuf[px[:, 1], px[:, 0]]
    px, L, P_, rad_ = px[out_], L[out_], P_[out_], rad_[out_]
    rbuf[px[:, 1], px[:, 0]] = rad_
    pos[px[:, 1], px[:, 0]] = P_
    nrm[px[:, 1], px[:, 0]] = L @ vnf[f]
    cov[px[:, 1], px[:, 0]] = True
P3 = pos[cov]
N3 = nrm[cov] / (np.linalg.norm(nrm[cov], axis=1)[:, None] + 1e-9)
# every photo's colour at every texel it sees, and how squarely it sees it
cols, scores = [], []
for v, (rv, t, logf, e) in enumerate(vs):
    img = cv2.imread(views[v][0]).astype(np.float32)
    R = rodrigues(rv)
    Xc = P3 @ R.T + t
    f = np.exp(logf)
    uv = f * Xc[:, :2] / Xc[:, 2:3] + np.array([cams[v]["w"] / 2, cams[v]["h"] / 2])
    facing = -(N3 @ R.T) @ np.array([0, 0, 1.0])
    # facing the camera, and not hidden behind the nose / cheek: a coarse depth test per pixel
    zb = np.full((img.shape[0] // 4 + 1, img.shape[1] // 4 + 1), np.inf)
    ij = np.clip((uv / 4).astype(int), 0, [zb.shape[1] - 1, zb.shape[0] - 1])
    vis = facing > 0.15
    np.minimum.at(zb, (ij[vis, 1], ij[vis, 0]), Xc[vis, 2])
    unhidden = Xc[:, 2] <= zb[ij[:, 1], ij[:, 0]] + 0.006
    inimg = (uv[:, 0] >= 0) & (uv[:, 1] >= 0) & (uv[:, 0] < img.shape[1] - 1) & (uv[:, 1] < img.shape[0] - 1)
    score = np.where(vis & unhidden & inimg, facing, 0).astype(np.float32)
    u_, v_ = np.clip(uv[:, 0], 0, img.shape[1] - 2), np.clip(uv[:, 1], 0, img.shape[0] - 2)  # bilinear sample
    x0_, y0_ = np.floor(u_).astype(int), np.floor(v_).astype(int)
    fx, fy = (u_ - x0_)[:, None], (v_ - y0_)[:, None]
    col = (img[y0_, x0_] * (1 - fx) * (1 - fy) + img[y0_, x0_ + 1] * fx * (1 - fy)
           + img[y0_ + 1, x0_] * (1 - fx) * fy + img[y0_ + 1, x0_ + 1] * fx * fy)
    cols.append(col)
    scores.append(score)
# each photo's brightness and colour matched to the front photo's where both see the skin well (generated views
# differ in exposure: without this the texture shows bands where one photo takes over from the next)
front = int(np.argmax([s_.sum() for s_ in scores]))
for v in range(len(cols)):
    if v == front:
        continue
    both = (scores[v] > 0.5) & (scores[front] > 0.5)
    if both.sum() > 500:
        # brightness only (one number), within limits: a per-channel match over-tints (the overlap is lit differently)
        lum = lambda c_: c_ @ np.array([0.114, 0.587, 0.299])  # noqa: E731  (BGR)
        gain = float(np.clip(np.median(lum(cols[front][both]) + 1) / np.median(lum(cols[v][both]) + 1), 0.8, 1.25))
        cols[v] = cols[v] * gain
        print("BRIGHTNESS MATCH", os.path.basename(views[v][0]), round(gain, 3))
# blended: each photo weighted by how squarely it sees the texel (sharply: the best view dominates, no seams)
W = np.array(scores) ** 6
sw = W.sum(0)
blend = (W[..., None] * np.array(cols)).sum(0) / np.maximum(sw, 1e-6)[:, None]
# the photo face stops under the chin: the neck in the photos is collar, shadow and background
lm_chin = lm[152]
chin_y = V[lm_chin, 1] if lm_chin >= 0 else -np.inf
keep = (sw > 1e-4) & (P3[:, 1] > chin_y - 0.012)
tex = np.zeros((tex_n, tex_n, 4), np.uint8)
ys, xs = np.where(cov)
tex[ys[keep], xs[keep], :3] = np.clip(blend[keep], 0, 255).astype(np.uint8)
tex[ys[keep], xs[keep], 3] = 255
tex = tex[::-1]  # image rows start at the top; the layout's v starts at the bottom
cv2.imwrite(os.path.join(out, "texture.png"), tex)
report["texture_photo_fraction"] = round(float((tex[..., 3] > 0).mean()), 3)
json.dump(report, open(os.path.join(out, "fit.json"), "w"), indent=1)
print("FLAME_PHOTOFIT_OK", out)
