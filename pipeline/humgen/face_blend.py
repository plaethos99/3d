"""Blend a baked photo face into a Human Generator skin texture (step 2b; pose-env python, -I).

Usage: python face_blend.py <face_dir> <base_texture.png | folder of HG skins> <out_texture.png> [--feather 14] [--tone face|body|detail] [--chroma 0.35]
       [--sigma 40] [--lips 0.75]
face_dir: face_bake.py's (bake_colour.png, bake_coverage.png, bake_hair.png if present).
Mask: where the photos saw the face (any coverage), not hair (hair mask), gaps closed and holes filled, shrunk a
little and feathered; around the eyes HG's own skin is kept (photo lids land a little off and show as rims).
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


candidates = None
if os.path.isdir(base_path):  # a folder of HG skins: the one closest in tone to the photo is picked below
    import glob
    candidates = sorted(p for p in glob.glob(os.path.join(base_path, "*.png"))
                        if os.path.basename(p)[:1].isupper() and "edited" not in p and "face_" not in p)
    base_path = candidates[0]
base = read(base_path, cv2.imread(base_path, cv2.IMREAD_UNCHANGED).shape[0])
n = base.shape[0]
base_rgb = base[..., :3][..., ::-1]
face = read(os.path.join(face_dir, "bake_colour.png"), n)
face_rgb, face_a = face[..., :3][..., ::-1], face[..., 3]
cov = read(os.path.join(face_dir, "bake_coverage.png"), n)[..., 0]
hair = read(os.path.join(face_dir, "bake_hair.png"), n)
k = max(3, int(round(feather * n / 4096)))
# Where the photos saw the face (any coverage at all: stills see the sides only at a slant), not hair. Hair is
# decided by the hair mask alone (a darkness rule took the eyebrows with it). Gaps (the nose's sides, around the
# mouth) are closed, holes (eyes, nostrils) filled: the face as one piece.
hairm = np.zeros((n, n), np.uint8)
if hair is not None:
    hairm = cv2.dilate((hair[..., 0] > 0.35).astype(np.uint8), np.ones((k, k), np.uint8))
hard = ((cov > 0.08) & (face_a > 0.5) & (hairm == 0)).astype(np.uint8)
hard = cv2.morphologyEx(hard, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def one_piece(h):
    cnt_, lab_, stats_, _ = cv2.connectedComponentsWithStats(h, 8)
    if cnt_ > 1:
        h = (lab_ == 1 + int(np.argmax(stats_[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    cs, _ = cv2.findContours(h, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return cv2.drawContours(np.zeros_like(h), cs, -1, 1, thickness=cv2.FILLED)


# hair the hair mask missed (strands at the temples, the hairline): darker than the skin and greyer, in big pieces
# (an opening wider than a brow is thick keeps the brows out of it)
lum_f = (face_rgb * [0.2126, 0.7152, 0.0722]).sum(2)
sat_f = (face_rgb.max(2) - face_rgb.min(2)) / np.maximum(face_rgb.max(2), 1e-3)
sk = hard > 0
if sk.sum() > 100:
    upper = np.zeros_like(sk)
    upper[:int(np.where(sk)[0].mean())] = True  # hair only comes in from above (the lips are dark and grey-ish too)
    hairish = (sk & upper & (lum_f < 0.6 * np.median(lum_f[sk])) & (sat_f < 0.8 * np.median(sat_f[sk]))).astype(np.uint8)
    # coloured hair (blue, green, purple...): skin always has more red than blue, hair like this doesn't. Thin
    # strands count too, so it skips the opening below (which keeps only big dark pieces)
    # (by colour direction: skin's red minus blue is large; any hair's - blue, green, grey-blue strands - is not)
    rb = face_rgb[..., 0] - face_rgb[..., 2]
    coloured = (sk & upper & (rb < 0.55 * np.median(rb[sk]))).astype(np.uint8)
    hairish = cv2.morphologyEx(hairish, cv2.MORPH_CLOSE, np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    hairish = cv2.morphologyEx(hairish, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6 * k + 1, 6 * k + 1)))
    hairish = cv2.dilate(hairish | cv2.morphologyEx(coloured, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)),
                         np.ones((2 * k + 1, 2 * k + 1), np.uint8))
    print("HAIR in the face area left out:", int(hairish.sum()), "texels")
    hard[hairish > 0] = 0
hard = one_piece(hard)
hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10 * k + 1, 10 * k + 1)))
hard = one_piece(hard)
hard[hairm > 0] = 0
hard = one_piece(cv2.morphologyEx(hard, cv2.MORPH_OPEN, np.ones((2 * k + 1, 2 * k + 1), np.uint8)))
hard = cv2.erode(hard, np.ones((k, k), np.uint8))
mask = cv2.GaussianBlur(hard.astype(np.float32), (0, 0), k)
inner = (hard > 0) & (cv2.erode(hard, np.ones((6 * k, 6 * k), np.uint8)) == 0)  # a band just inside the edge
fl = to_lin(face_rgb)
if inner.sum() < 100:
    raise SystemExit("FACE_BLEND_FAILED: almost no face found in the bake")
if candidates:  # closest skin: brightness and colour (log ratios) of the face's edge band, photo vs each skin
    want = np.log(np.median(fl[inner], axis=0) + 1e-4)
    scores = []
    for cpath in candidates:
        c = read(cpath, n)[..., :3][..., ::-1]
        have = np.log(np.median(to_lin(c)[inner], axis=0) + 1e-4)
        d = want - have
        # mostly the skin's colour (channel ratios); brightness counts little: it is mostly the photo's lighting
        scores.append((float(0.25 * np.abs(d.mean()) + 2.0 * np.abs(d - d.mean()).sum()), cpath, c))
    scores.sort(key=lambda t: t[0])
    print("SKINS", [(os.path.basename(p), round(sc, 3)) for sc, p, _ in scores])
    _, base_path, base_rgb = scores[0]
    base = read(base_path, n)
    print("SKIN_PICKED", os.path.basename(base_path))
bl = to_lin(base_rgb)
# around the eyes HG's own skin stays: the photo's lids land a little off HG's and show as pink or dark rims.
# The eye openings are the two biggest darker pieces of HG's texture in the upper half of the face.
lum_b = (base_rgb * [0.2126, 0.7152, 0.0722]).sum(2)
dark = ((lum_b < 0.85 * np.median(lum_b[hard > 0])) & (hard > 0)).astype(np.uint8)
dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
cnt_d, lab_d, st_d, cen_d = cv2.connectedComponentsWithStats(dark, 8)
face_cy = np.where(hard > 0)[0].mean()
eyes = [i for i in range(1, cnt_d) if cen_d[i][1] < face_cy and st_d[i, cv2.CC_STAT_AREA] > 4 * k * k]
eyes = sorted(eyes, key=lambda i: -st_d[i, cv2.CC_STAT_AREA])[:2]
if eyes:
    zone = cv2.dilate(np.isin(lab_d, eyes).astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5 * k + 1, 5 * k + 1)))
    mask = mask * (1 - cv2.GaussianBlur(zone.astype(np.float32), (0, 0), k))
    print("EYES kept from HG:", len(eyes), "areas", [int(st_d[i, cv2.CC_STAT_AREA]) for i in eyes])
# the nostrils too: the photo's dark nostrils seldom land on HG's and show as torn brown patches. HG's are its two
# darker pieces below the eyes, near the middle
if eyes:
    eye_y = max(cen_d[i][1] for i in eyes)
    mid_x = np.mean([cen_d[i][0] for i in eyes])
    span = abs(cen_d[eyes[0]][0] - cen_d[eyes[-1]][0]) if len(eyes) == 2 else 20 * k
    nos = [i for i in range(1, cnt_d) if i not in eyes and cen_d[i][1] > eye_y + 0.3 * span
           and abs(cen_d[i][0] - mid_x) < 0.35 * span and cen_d[i][1] < eye_y + 0.85 * span
           and st_d[i, cv2.CC_STAT_AREA] > k * k // 2]
    if nos:
        nz = np.isin(lab_d, nos).astype(np.uint8)
        if len(nos) == 1:  # only one found: mirror it across the face's middle so both nostrils stay HG's
            nz = nz | cv2.flip(nz, 1) if abs(cen_d[nos[0]][0] - mid_x) > k else nz
        nz = cv2.dilate(nz, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (6 * k + 1, 4 * k + 1)))
        mask = mask * (1 - cv2.GaussianBlur(nz.astype(np.float32), (0, 0), k))
        print("NOSTRILS kept from HG:", len(nos))
# a "keep the base's own skin here" mask from the caller (Genesis 9: its eyes, nostrils and lips, from its landmarks),
# applied after the face mask was made one piece (that fills holes)
kp = os.path.join(face_dir, "bake_keep.png")
if os.path.exists(kp):
    keep = read(kp, n)
    keep = keep[..., 0] if keep.ndim == 3 else keep
    mask = mask * (1 - cv2.GaussianBlur(np.clip(keep, 0, 1).astype(np.float32), (0, 0), k))
    print("KEEP base skin where the caller said:", round(float((keep > 0.5).mean()) * 100, 2), "% of the texture")
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
    # finer than the photo's lighting (shadows under the eyes, beside the mouth) but coarser than brows, lips, creases
    sig = float(sys.argv[sys.argv.index("--sigma") + 1]) if "--sigma" in sys.argv else 40.0
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
if eyes:  # HG's skin kept around the eyes takes the colour of the face just around it (else a pink or pale ring)
    zm = cv2.GaussianBlur(zone.astype(np.float32), (0, 0), k)
    ring = (cv2.dilate(zone, np.ones((4 * k + 1, 4 * k + 1), np.uint8)) > 0) & (zone == 0) & (hard > 0)
    lid = (zone > 0) & (lum_b > 0.85 * np.median(lum_b[hard > 0]))  # the zone's skin, not the eye openings
    if ring.sum() > 100 and lid.sum() > 100:
        g_eye = np.median(outl[ring], axis=0) / np.maximum(np.median(outl[lid], axis=0), 1e-4)
        g_eye = 1 + 0.8 * (np.clip(g_eye, 0.5, 2.0) - 1)
        print("EYE ZONE colour matched to the face around it:", np.round(g_eye, 3).tolist())
        outl = outl * (1 + (g_eye - 1) * zm[..., None])
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
