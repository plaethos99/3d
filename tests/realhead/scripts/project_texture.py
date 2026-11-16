"""Paint the head's colour texture from video frames, the way texture projection works in 3D tools.

Usage: project_texture.py <bake_dir> <texture_cameras.json> <clips_dir> <subject_fit.json> <out_dir>

Every texel is looked up in every frame that can see it. Frames that face that spot squarely count
the most. The background and the hoodie are masked out. The side of the face never turned toward
the camera borrows colour from the mirrored side. Whatever no frame saw, such as the back of the
head and the neck under the hoodie, is filled in smoothly from its surroundings.
"""
import json
import os
import sys

import cv2
import numpy as np

bake_dir, cams_path, clips, fit_path, out_dir = sys.argv[1:6]
os.makedirs(out_dir, exist_ok=True)
position = np.load(os.path.join(bake_dir, "uv_position.npy")).astype(np.float32)
normal = np.load(os.path.join(bake_dir, "uv_normal.npy")).astype(np.float32)
covered = np.load(os.path.join(bake_dir, "uv_covered.npy"))
with open(cams_path, encoding="utf-8") as f:
    views = json.load(f)
with open(fit_path, encoding="utf-8") as f:
    fit = json.load(f)
size = position.shape[0]
# Hair rules below are written for an adult head with eyes 64.6 mm apart; HEAD_SCALE adapts them.
HEAD_SCALE = float(os.environ.get("HEAD_SCALE", "1.0"))
# Optional: the tracked face points for each image, to keep the eyes out of the skin texture.
EYE_CONTOURS = [[33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
                [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]]
eye_records = json.load(open(os.environ["EYE_MASK_VIEWS"], encoding="utf-8")) if os.environ.get("EYE_MASK_VIEWS") else []
HAIR_TONE = os.environ.get("HAIR_TONE", "dark")  # "dark" hair, or "light" (grey or white) hair
FACING_POWER = float(os.environ.get("FACING_POWER", "10"))
index = {i: k for k, i in enumerate(fit["ids"])}
chin_z = float(np.array(fit["points"])[index[152]][2])


def frame_image(video, frame):
    """A video frame, or a still image with the same name. Stills may carry a transparency mask."""
    still = os.path.join(clips, video + ".png")
    if os.path.exists(still):
        image = cv2.imread(still, cv2.IMREAD_UNCHANGED)
        if image.ndim == 3 and image.shape[2] == 4:
            return image[:, :, :3], image[:, :, 3]
        return image, None
    capture = cv2.VideoCapture(os.path.join(clips, video + ".mp4"))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, bgr = capture.read()
    capture.release()
    return bgr, None


def subject_mask(bgr, alpha=None):
    """1 where the pixel shows the subject. Uses the image's transparency when it has one;
    otherwise drops the pink backdrop and the white hoodie of the video this was built for."""
    if alpha is not None:
        return cv2.erode((alpha > 128).astype(np.float32), np.ones((5, 5), np.uint8))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hue, sat, val = hsv[:, :, 0] * 2, hsv[:, :, 1] / 255, hsv[:, :, 2] / 255
    # His skin and hair are darker than the backdrop and the hoodie, and warmer than both.
    dark = val < 0.55
    warm = (hue < 40) & (val < 0.8) & (sat > 0.2)
    grey = (sat < 0.15) & (val > 0.35)  # hoodie folds in shadow
    mask = ((dark | warm) & ~grey).astype(np.float32)
    # Stay a few pixels clear of the edge, where colours mix.
    return cv2.erode(mask, np.ones((7, 7), np.uint8))


def gather(points, normals, weight_scale):
    colour = np.zeros((size, size, 3), np.float32)
    weight = np.zeros((size, size), np.float32)
    flat_p = points.reshape(-1, 3)
    for key, view in views.items():
        seen = np.load(os.path.join(bake_dir, f"seen_{key}.npy")).astype(np.float32)
        if weight_scale < 1.0:
            seen = seen[:, ::-1] * 0 + 1.0  # mirrored lookups test facing only; see below
        p = np.array(view["projection"])
        homog = np.hstack([flat_p, np.ones((len(flat_p), 1), np.float32)]) @ p.T
        uv = (homog[:, :2] / homog[:, 2:]).reshape(size, size, 2).astype(np.float32)
        centre = np.array(view["centre"], np.float32)
        to_cam = centre - points
        to_cam /= np.linalg.norm(to_cam, axis=2, keepdims=True) + 1e-9
        facing = np.clip((normals * to_cam).sum(2), 0, 1)
        bgr, alpha = frame_image(view["video"], view["frame"])
        mask = subject_mask(bgr, alpha)
        sample = cv2.remap(bgr.astype(np.float32), uv[:, :, 0], uv[:, :, 1], cv2.INTER_LINEAR, borderValue=(0, 0, 0))
        if eye_records:
            # Inside the eye opening the photo shows the eye itself, not skin: keep it off the skin.
            record = next((r for r in eye_records if r["video"] == view["video"] and r["frame"] == view["frame"]), None)
            opening_file = os.path.join(os.environ.get("EYE_OPENING_MASKS", ""), f"eyes_{view['video']}.png")
            if os.environ.get("EYE_OPENING_MASKS") and os.path.exists(opening_file):
                # Openings cut from the image itself; the tracker's outline is too small on cartoon eyes.
                opening = cv2.imread(opening_file, 0) > 127
                mask[cv2.dilate(opening.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0] = 0.0
            elif record is not None:
                marks = np.array(record["points"])[:, :2]
                for ids in EYE_CONTOURS:
                    cv2.fillPoly(mask, [marks[ids].astype(np.int32)], 0.0)
        inside = cv2.remap(mask, uv[:, :, 0], uv[:, :, 1], cv2.INTER_LINEAR, borderValue=0)
        # A high power lets the most squarely facing frame dominate, so blending doesn't blur detail.
        w = (facing ** FACING_POWER) * inside * covered * weight_scale
        if weight_scale >= 1.0:
            w *= seen
        colour += sample * w[:, :, None]
        weight += w
    return colour, weight


direct_colour, direct_weight = gather(position, normal, 1.0)
# Mirrored: look up the point's reflection, which the frames saw from the other side.
# Visibility of the reflected point isn't baked, so facing alone gates it, at a lower trust.
mirror_p = position * np.array([-1, 1, 1], np.float32)
mirror_n = normal * np.array([-1, 1, 1], np.float32)
mirror_colour, mirror_raw = gather(mirror_p, mirror_n, 0.999)
direct = direct_colour / np.maximum(direct_weight, 1e-6)[:, :, None]
mirrored = mirror_colour / np.maximum(mirror_raw, 1e-6)[:, :, None]
# The two sides were lit differently, so a mirrored patch can come out lighter or darker than the
# real side next to it. Where both exist, match the mirrored colour's brightness to the real one.
both = ((direct_weight > 0.02) & (mirror_raw > 0.02)).astype(np.float32)
spread = lambda a: cv2.GaussianBlur(a, (0, 0), 40)
gain = spread(direct * both[:, :, None]) / np.maximum(spread(mirrored * both[:, :, None]), 1e-3)
gain = np.where(spread(both)[:, :, None] > 0.02, np.clip(gain, 0.6, 1.6), 1.0)
mirrored = mirrored * gain
# Hand over from real to mirrored colour gradually, not at a hard line.
need = np.clip(1.0 - direct_weight / 0.08, 0, 1)
need = np.maximum(cv2.GaussianBlur(need, (0, 0), 10), need * (direct_weight < 1e-4))
mirror_weight = mirror_raw * 0.5 * need
colour = direct * (1 - need[:, :, None]) + mirrored * need[:, :, None]
colour = np.where((mirror_raw < 1e-4)[:, :, None], direct, colour)
total = direct_weight * (1 - need) + mirror_weight

# The neck under the hoodie was never seen: leave it to the fill below.
rule_position = position / HEAD_SCALE
below_chin = position[:, :, 2] < chin_z - 0.012 * HEAD_SCALE
total[below_chin & (position[:, :, 1] < 0.02)] *= 0.0
known = (total > 0.004) & covered
print("TEXELS_FROM_PHOTOS", f"{known.sum() / covered.sum() * 100:.0f}% of the head",
      "of which mirrored", f"{((mirror_weight > direct_weight) & known).sum() / covered.sum() * 100:.0f}%")

# Fill unseen texels: average the known colour outward, coarse to fine (a push-pull fill).
filled = colour.copy()
filled[~known] = 0
weights = known.astype(np.float32)
levels = []
c, w = filled * weights[:, :, None], weights
while c.shape[0] > 8:
    levels.append((c, w))
    c = cv2.resize(c, (c.shape[1] // 2, c.shape[0] // 2), interpolation=cv2.INTER_AREA) * 4
    w = cv2.resize(w, (w.shape[1] // 2, w.shape[0] // 2), interpolation=cv2.INTER_AREA) * 4
estimate = c / np.maximum(w, 1e-6)[:, :, None]
for c, w in reversed(levels):
    up = cv2.resize(estimate, (c.shape[1], c.shape[0]), interpolation=cv2.INTER_LINEAR)
    own = c / np.maximum(w, 1e-6)[:, :, None]
    blend = np.clip(w, 0, 1)[:, :, None]
    estimate = own * blend + up * (1 - blend)
result = np.where(known[:, :, None], colour, estimate)

# Regions no frame saw get the colour they must be: hair on the back and top of the head, skin on
# the neck. Each takes the median of what the photos did show of it, blended in softly.
seen_hair = known & (rule_position[:, :, 2] > 0.075) & (rule_position[:, :, 1] < 0.06)
seen_skin = known & (np.abs(rule_position[:, :, 0]) > 0.035) & (rule_position[:, :, 2] < -0.02) & (position[:, :, 2] > chin_z)
hair_colour = np.median(colour[seen_hair], axis=0)
skin_colour = np.median(colour[seen_skin], axis=0)
print("HAIR_BGR", hair_colour.round(1), "SKIN_BGR", skin_colour.round(1))
# The hairline: behind the ears it stops about level with the ear lobe; only at the middle of the
# back does it dip lower, to the nape. (Measured where the photos show it; otherwise typical.)
behind = np.clip((rule_position[:, :, 1] - 0.02) / 0.04, 0, 1)
back_middle = np.clip((rule_position[:, :, 1] - 0.10) / 0.06, 0, 1) * np.clip(1 - np.abs(rule_position[:, :, 0]) / 0.06, 0, 1)
hair_line = -0.03 - 0.008 * behind - 0.04 * back_middle
hairy = np.clip((rule_position[:, :, 2] - hair_line) / 0.015, 0, 1) * np.clip((rule_position[:, :, 1] + 0.01) / 0.03, 0, 1)
hairy = np.maximum(hairy, np.clip((rule_position[:, :, 2] - 0.085) / 0.02, 0, 1))
# Where no photo shows the hairline, give it a natural, slightly uneven edge rather than a ruler line.
rng = np.random.default_rng(5)
wobble = cv2.GaussianBlur(rng.standard_normal((size, size)).astype(np.float32), (0, 0), 25)
wobble = wobble / max(float(wobble.std()), 1e-6) * 0.004
# Behind the face only (y beyond 15 mm back), so eye sockets and cheeks never count as scalp.
behind_face = np.clip((rule_position[:, :, 1] - 0.015) / 0.02, 0, 1)
hairy = np.maximum(np.clip((rule_position[:, :, 2] + wobble - hair_line) / 0.008, 0, 1) * behind_face,
                   np.clip((rule_position[:, :, 2] - 0.085) / 0.02, 0, 1))
target = hair_colour * hairy[:, :, None] + skin_colour * (1 - hairy[:, :, None])
unseen = np.clip(1.0 - total / 0.05, 0, 1) * covered
# Small unseen patches inside seen skin (a crease every photo hid, say) take their colour from the
# skin around them; only large unseen areas (the back of the head) take the flat hair/skin colour.
count, pieces, stats, _ = cv2.connectedComponentsWithStats(((unseen > 0.5) & covered).astype(np.uint8))
small = np.zeros_like(covered)
for k in range(1, count):
    if stats[k, cv2.CC_STAT_AREA] < 0.004 * covered.sum():
        small |= pieces == k
small_soft = np.maximum(cv2.GaussianBlur(cv2.dilate(small.astype(np.uint8), np.ones((9, 9), np.uint8)).astype(np.float32), (0, 0), 3), small)
target = target * (1 - small_soft[:, :, None]) + estimate * small_soft[:, :, None]
# Widen the unseen weight around small patches too, so their edge doesn't show as an outline.
unseen = np.maximum(unseen, small_soft * covered)
print("SMALL_UNSEEN_PATCHES", int(sum(1 for k in range(1, count) if stats[k, cv2.CC_STAT_AREA] < 0.004 * covered.sum())))
result = result * (1 - unseen[:, :, None]) + target * unseen[:, :, None]

# Where the photos do show it, the hairline is where his skin turns to dark hair. Only the scalp
# counts: his beard and eyebrows are dark too, but they aren't part of the scalp hair.
luminance = cv2.cvtColor(np.clip(colour, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
if HAIR_TONE == "light":
    # Grey or white hair: bright and colourless, unlike skin, which is warmer.
    hsv = cv2.cvtColor(np.clip(colour, 0, 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    dark = np.clip((0.22 - hsv[:, :, 1] / 255) / 0.1, 0, 1) * np.clip((luminance - 110.0) / 30.0, 0, 1)
else:
    dark = np.clip((40.0 - luminance) / 16.0, 0, 1)
sideburns = (np.abs(rule_position[:, :, 0]) > 0.064) & (rule_position[:, :, 1] > -0.005) & (rule_position[:, :, 2] > -0.015)
# In front of the back of the ear, scalp hair stops above the jaw; his beard is not scalp hair.
above_jaw = np.where(rule_position[:, :, 1] < 0.065, np.clip((rule_position[:, :, 2] + 0.035) / 0.008, 0, 1), 1.0)
possible = np.maximum.reduce([np.clip((rule_position[:, :, 2] - (hair_line - 0.005)) / 0.008, 0, 1) * behind_face * above_jaw,
                              np.clip((rule_position[:, :, 2] - 0.055) / 0.01, 0, 1), sideburns.astype(np.float32)])
seen_share = np.clip(total / 0.05, 0, 1)
hairy = cv2.GaussianBlur(possible * dark, (0, 0), 1.5) * seen_share + hairy * (1 - seen_share)
# The ears and eye sockets have their own separate pieces of the texture. A piece that sits mostly
# in the ear or eye area is not scalp, however dark it is, so it gets no hair at all.
pieces, piece_of = cv2.connectedComponents(covered.astype(np.uint8))
for piece in range(1, pieces):
    where = piece_of == piece
    centre = position[where].mean(axis=0) / HEAD_SCALE
    ear_piece = abs(centre[0]) > 0.068 and -0.07 < centre[2] < 0.05 and 0.03 < centre[1] < 0.14
    eye_piece = 0.012 < abs(centre[0]) < 0.055 and abs(centre[2]) < 0.03 and centre[1] < 0.04
    if (ear_piece or eye_piece) and where.sum() < covered.sum() * 0.2:
        hairy[where] = 0.0

# Soften the seam where photo colour meets filled colour.
edge = cv2.GaussianBlur(known.astype(np.float32), (0, 0), 6)
result = result * edge[:, :, None] + cv2.GaussianBlur(result, (0, 0), 3) * (1 - edge[:, :, None])
# Repaint the small unseen patches (and a rim around them, where the photos' colour is unreliable
# shadow) from the skin around them, so they leave no mark.
rim = cv2.dilate(small.astype(np.uint8), np.ones((11, 11), np.uint8))
result = cv2.inpaint(np.clip(result, 0, 255).astype(np.uint8), rim, 6, cv2.INPAINT_TELEA).astype(np.float32)
cv2.imwrite(os.path.join(out_dir, "skin_colour.png"), np.clip(result, 0, 255).astype(np.uint8))
grey = cv2.cvtColor(np.clip(result, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
detail = grey - cv2.GaussianBlur(grey, (0, 0), 4)
cv2.imwrite(os.path.join(out_dir, "skin_detail.png"), np.clip(128 + detail * 3, 0, 255).astype(np.uint8))
cv2.imwrite(os.path.join(out_dir, "hair_mask.png"), (np.clip(hairy, 0, 1) * 255).astype(np.uint8))
cv2.imwrite(os.path.join(out_dir, "photo_coverage.png"), (np.clip(total / 1.0, 0, 1) * 255).astype(np.uint8))
print("TEXTURE_OK")
