"""Blend a baked photo face into a Human Generator skin texture (step 2b; pose-env python, -I).

Usage: python face_blend.py <face_dir> <base_texture.png> <out_texture.png> [--feather 14] [--tone face|body|detail] [--chroma 0.35]
       [--sigma 90] [--lips 0.75]
face_dir: face_bake.py's (bake_colour.png, bake_coverage.png, bake_hair.png if present).
Mask: where the photos saw skin (coverage), not hair, shrunk a little and feathered.
Tone (one skin tone everywhere): with --tone face (default) HG's whole texture is shifted, per colour channel in
linear light, so its skin just inside the face edge matches the photo's there (the person's own tone):
brightness fully, colour by --chroma (a photo's colour cast is mostly its lighting); with
--tone body the photo face is shifted to HG's tone instead; with --tone detail the photo keeps only its detail
(finer than --sigma texels at 4K) and takes HG's broad colour under it: no patch, HG's skin tone. The lips
(the reddest piece of HG's face texture) take HG's lip colour at a fine scale (--lips: how much). Then face over body with the feathered mask.
Writes the texture (same size as the base) and <face_dir>/blend_check.jpg (mask, before, after around the head).
Prints FACE_BLEND_OK.
"""
import os
import sys

import cv2
import numpy as np

face_dir, base_path, out_path = (os.path.abspath(p) for p in sys.argv[1:4])
feather = float(sys.argv[sys.argv.index("--feather") + 1]) if "--feather" in sys.argv else 14.0
tone = sys.argv[sys.argv.index("--tone") + 1] if "--tone" in sys.argv else "face"
chroma = float(sys.argv[sys.argv.index("--chroma") + 1]) if "--chroma" in sys.argv else 0.35


def to_lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def read(path, n):
    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.shape[0] != n:
        im = cv2.resize(im, (n, n), interpolation=cv2.INTER_AREA)
    return im.astype(np.float32) / (65535.0 if im.dtype == np.uint16 else 255.0)


base = read(base_path, cv2.imread(base_path, cv2.IMREAD_UNCHANGED).shape[0])
n = base.shape[0]
base_rgb = base[..., :3][..., ::-1]
face = read(os.path.join(face_dir, "bake_colour.png"), n)
face_rgb, face_a = face[..., :3][..., ::-1], face[..., 3]
cov = read(os.path.join(face_dir, "bake_coverage.png"), n)[..., 0]
hair = read(os.path.join(face_dir, "bake_hair.png"), n)
m = np.clip((cov - 0.3) / 0.4, 0, 1) * (face_a > 0.5)
if hair is not None:
    m *= 1 - np.clip((hair[..., 0] - 0.2) / 0.5, 0, 1)
k = max(3, int(round(feather * n / 4096)))
hard = (m > 0.5).astype(np.uint8)
hard = cv2.morphologyEx(hard, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
# stray hair at the edge: texels much darker than the face's skin are dropped (brows, lashes and other dark
# spots inside the face come back below, when the holes are filled)
lum = (face_rgb * [0.2126, 0.7152, 0.0722]).sum(2)
hard[(lum < 0.6 * np.median(lum[hard > 0]))] = 0
hard = cv2.morphologyEx(hard, cv2.MORPH_OPEN, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
# the face only: its largest piece, holes (eyes, mouth slit, nostrils, brows) filled
cnt, lab, stats, _ = cv2.connectedComponentsWithStats(hard, 8)
if cnt > 1:
    hard = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
contours, _ = cv2.findContours(hard, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
hard = cv2.drawContours(np.zeros_like(hard), contours, -1, 1, thickness=cv2.FILLED)
hard = cv2.erode(hard, np.ones((k, k), np.uint8))
mask = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), k)
inner = (hard > 0) & (cv2.erode(hard, np.ones((6 * k, 6 * k), np.uint8)) == 0)  # a band just inside the edge
bl, fl = to_lin(base_rgb), to_lin(face_rgb)
if inner.sum() < 100:
    raise SystemExit("FACE_BLEND_FAILED: almost no face found in the bake")
gain = np.median(fl[inner], axis=0) / np.maximum(np.median(bl[inner], axis=0), 1e-4)
# brightness fully, colour only partly: a photo's colour cast is mostly its light, not the skin
W = np.array([0.2126, 0.7152, 0.0722])
g_lum = float((np.median(fl[inner], axis=0) * W).sum() / max((np.median(bl[inner], axis=0) * W).sum(), 1e-4))
gain = g_lum * (gain / g_lum) ** chroma
print("TONE gain (photo / HG, linear R G B):", np.round(gain, 3).tolist(), "band texels", int(inner.sum()))
if tone == "face":
    bl = bl * gain
elif tone == "body":
    fl = fl / gain
else:  # "detail": the photo's detail on HG's own broad colour: no patch, one skin tone (HG's texture choice)
    sig = float(sys.argv[sys.argv.index("--sigma") + 1]) if "--sigma" in sys.argv else 90.0
    sig *= n / 4096
    M = (hard > 0).astype(np.float32)
    Mb = cv2.GaussianBlur(M, (0, 0), sig) + 1e-4
    low_b = cv2.GaussianBlur(bl * M[..., None], (0, 0), sig) / Mb[..., None]
    low_f = cv2.GaussianBlur(fl * M[..., None], (0, 0), sig) / Mb[..., None]
    fl_detail = fl * np.clip(low_b / np.maximum(low_f, 1e-4), 0.2, 5.0)
    # lips: the broad colour washes them out; there, take the colour at a fine scale (lip-sized) from HG's lips
    lips_amount = float(sys.argv[sys.argv.index("--lips") + 1]) if "--lips" in sys.argv else 0.75
    red = base_rgb[..., 0] - base_rgb[..., 1]
    face_red = red[hard > 0]
    thr = np.median(face_red) + 4 * np.median(np.abs(face_red - np.median(face_red)))
    lips = ((red > thr) & (hard > 0)).astype(np.uint8)
    lips = cv2.morphologyEx(lips, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    cnt, lab, stats, cen = cv2.connectedComponentsWithStats(lips, 8)
    if cnt > 1:  # the lips: the biggest red piece
        lips = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
        lips = cv2.dilate(lips, np.ones((k, k), np.uint8))
        lip_m = cv2.GaussianBlur(lips.astype(np.float32), (0, 0), k / 2) * lips_amount
        s2 = 10 * n / 4096
        Lb = cv2.GaussianBlur(bl * M[..., None], (0, 0), s2) / (cv2.GaussianBlur(M, (0, 0), s2) + 1e-4)[..., None]
        Lf = cv2.GaussianBlur(fl * M[..., None], (0, 0), s2) / (cv2.GaussianBlur(M, (0, 0), s2) + 1e-4)[..., None]
        fl_lips = fl * np.clip(Lb / np.maximum(Lf, 1e-4), 0.2, 5.0)
        fl = fl_detail * (1 - lip_m[..., None]) + fl_lips * lip_m[..., None]
        print("LIPS", int(lips.sum()), "texels, colour from HG's lips at", lips_amount)
    else:
        fl = fl_detail
outl = bl * (1 - mask[..., None]) + fl * mask[..., None]
out = to_srgb(outl)
res = np.concatenate([out[..., ::-1], base[..., 3:4]], axis=2) if base.shape[2] == 4 else out[..., ::-1]
cv2.imwrite(out_path, (np.clip(res, 0, 1) * 255 + 0.5).astype(np.uint8))
ys, xs = np.where(hard > 0)
y0, y1, x0, x1 = max(ys.min() - 3 * k, 0), min(ys.max() + 3 * k, n), max(xs.min() - 3 * k, 0), min(xs.max() + 3 * k, n)


def crop(a):
    a = a[y0:y1, x0:x1]
    return cv2.resize(a, (int(600 * a.shape[1] / a.shape[0]), 600))


check = np.hstack([crop(np.dstack([mask] * 3)), crop(base_rgb[..., ::-1]), crop(out[..., ::-1])])
cv2.imwrite(os.path.join(face_dir, "blend_check.jpg"), (check * 255).astype(np.uint8))
print("FACE_BLEND_OK", out_path, f"face area {hard.mean() * 100:.1f}% of the texture")
