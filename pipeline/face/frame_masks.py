"""Hair, skin and backdrop in each chosen frame of a head clip (for carving the hair and keeping it off the skin).

Usage: python frame_masks.py <texture_cameras.json> <views.json> <clips_dir> <out_dir>
texture_cameras.json: solve_view_cameras.py's (which frames); views.json: scan_views.py's (478 tracked points
per frame); clips_dir: where <video>.mp4 is.
Per frame, written as <key>_fg.png, <key>_hair.png, <key>_skin.png (white = yes):
- fg: the subject, everything that differs from the backdrop (the picture's border colour, in Lab), above
  the clothes (nothing lower than the chin by more than a quarter of the face's height);
- skin and hair: each subject pixel goes to the nearer of two colour models (Lab, Mahalanobis distance),
  learned in this frame: skin from the cheeks and forehead (inside the tracked face, clear of eyes, brows
  and lips), hair from just above the forehead's top and beside the temples, where hair is if the person
  has any. Inside the tracked face outline it is skin (eyebrows and lashes aren't hair here), and so is the
  neck under the jaw (between the jaw's corners, below the chin), however shadowed; hair is the one mass
  joined to its biggest piece.
Also hair.json: the hair's median colour (sRGB, 0-255) over all frames, and each frame's hair share.
"""
import json
import os
import sys

import cv2
import numpy as np

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
             176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
CHEEKS_FOREHEAD = [[117, 118, 101, 36, 205, 187, 123], [346, 347, 330, 266, 425, 411, 352],
                   [67, 109, 10, 338, 297, 299, 9, 69]]


def frame(clips, video, index):
    still = os.path.join(clips, video + ".png")  # a still (face_from_image --views stills) instead of a clip
    if os.path.exists(still):
        return cv2.imread(still)
    cap = cv2.VideoCapture(os.path.join(clips, video + ".mp4"))
    cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, bgr = cap.read()
    cap.release()
    return bgr


def model(lab, mask):
    px = lab[mask > 0].reshape(-1, 3)
    mean = px.mean(0)
    cov = np.cov(px.T) + np.eye(3) * 4.0
    return mean, np.linalg.inv(cov)


def mahal(lab, m):
    d = lab.reshape(-1, 3) - m[0]
    return np.einsum("ij,jk,ik->i", d, m[1], d).reshape(lab.shape[:2])


def main():
    cams_path, views_path, clips, out = sys.argv[1:5]
    os.makedirs(out, exist_ok=True)
    cams = json.load(open(cams_path, encoding="utf-8"))
    views = json.load(open(views_path, encoding="utf-8"))
    hair_px, report = [], {}
    for key, cam in cams.items():
        rec = next(r for r in views if r["video"] == cam["video"] and r["frame"] == cam["frame"])
        marks = np.array(rec["points"])[:, :2]
        bgr = frame(clips, cam["video"], cam["frame"])
        h, w = bgr.shape[:2]
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        border = np.concatenate([lab[:6].reshape(-1, 3), lab[-6:].reshape(-1, 3), lab[:, :6].reshape(-1, 3),
                                 lab[:, -6:].reshape(-1, 3)])
        fg = (np.linalg.norm(lab - np.median(border, 0), axis=2) > 22).astype(np.uint8)
        chin, brow = marks[152, 1], marks[10, 1]
        face_h = chin - brow
        fg[int(max(chin + 0.25 * face_h, 0)):] = 0
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        # colour seeds
        skin_seed = np.zeros((h, w), np.uint8)
        for ids in CHEEKS_FOREHEAD:
            cv2.fillPoly(skin_seed, [marks[ids].astype(np.int32)], 1)
        skin_seed &= fg
        hair_seed = np.zeros((h, w), np.uint8)
        top = marks[10]
        eye_w = np.linalg.norm(marks[33] - marks[263])
        for c, r in ((top, 0.25), (marks[54], 0.18), (marks[284], 0.18)):  # above the forehead, the temples
            centre = c + np.array([0, -0.45 * eye_w])
            cv2.circle(hair_seed, tuple(np.round(centre).astype(int)), max(int(r * eye_w), 3), 1, -1)
        hair_seed &= fg
        face = np.zeros((h, w), np.uint8)
        cv2.fillPoly(face, [marks[FACE_OVAL].astype(np.int32)], 1)
        hair_seed[face > 0] = 0
        skin_m = model(lab, skin_seed)
        if hair_seed.sum() > 50 and np.linalg.norm(lab[hair_seed > 0].mean(0) - skin_m[0]) > 12:
            hair_m = model(lab, hair_seed)
            hair = (mahal(lab, hair_m) < mahal(lab, skin_m)) & (fg > 0) & (face == 0)
        else:  # no hair (or hair as light as skin): none
            hair = np.zeros((h, w), bool)
        hair = cv2.morphologyEx(hair.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        hair = cv2.morphologyEx(hair, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8)) & fg
        # the neck under the jaw is skin, however shadowed: below the chin, between the jaw's corners
        jaw = sorted((marks[172, 0], marks[397, 0]))
        inset = 0.12 * (jaw[1] - jaw[0])
        hair[int(chin):, int(jaw[0] + inset):int(jaw[1] - inset)] = 0
        # hair is one mass: only the pieces joined to the biggest one (stray dark spots go)
        n, lab_ids, stats, _ = cv2.connectedComponentsWithStats(hair, 8)
        if n > 2:
            big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            hair = (lab_ids == big).astype(np.uint8)
        skin = (fg > 0) & (hair == 0)
        cv2.imwrite(os.path.join(out, f"{key}_fg.png"), fg * 255)
        cv2.imwrite(os.path.join(out, f"{key}_hair.png"), hair * 255)
        cv2.imwrite(os.path.join(out, f"{key}_skin.png"), skin.astype(np.uint8) * 255)
        if hair.sum():
            hair_px.append(bgr[cv2.erode(hair, np.ones((5, 5), np.uint8)) > 0])
        report[key] = round(float(hair.sum() / max(fg.sum(), 1)), 3)
        # a picture to check by eye: hair green, skin red, over the frame
        view = bgr.copy()
        view[hair > 0] = (0.5 * view[hair > 0] + (0, 127, 0)).astype(np.uint8)
        view[skin] = (0.6 * view[skin] + (0, 0, 100)).astype(np.uint8)
        cv2.imwrite(os.path.join(out, f"{key}_check.jpg"), view)
    colour = np.median(np.concatenate(hair_px), 0)[::-1].tolist() if hair_px else None
    json.dump({"hair_rgb": colour, "hair_share": report}, open(os.path.join(out, "hair.json"), "w"), indent=1)
    print("MASKS_OK", out, "hair colour", None if colour is None else [round(c) for c in colour], report)


if __name__ == "__main__":
    main()
