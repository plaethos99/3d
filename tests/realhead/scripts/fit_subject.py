"""Measure the subject's face in 3D from many video frames, starting from the base head's points.

Usage: fit_subject.py <anchors.json> <base_views_dir> <views.json> <out.json>

Alternates two steps, like a photogrammetry solve:
  1. With the face's 3D points fixed, find where the camera was for each frame (perspective-n-point).
  2. With the cameras fixed, move each 3D point to where the frames that see it squarely put it.
Points that the tracker places inconsistently even on the base head's own renders, such as the
face's outline, count for less. Overall size can't be measured from photos without a known
object in view, so it stays the base head's adult size; proportions come from the photos.
"""
import json
import os
import sys

import cv2
import numpy as np

anchors_path, base_dir, views_path, out_path = sys.argv[1:5]
# Squared weights of the pull toward the base head, in the picture and along the line of sight.
PRIOR_PLANE = float(os.environ.get("PRIOR_PLANE", "3000")) ** 2
PRIOR_DEPTH = float(os.environ.get("PRIOR_DEPTH", "3000")) ** 2
with open(anchors_path, encoding="utf-8") as f:
    anchors = json.load(f)
with open(os.path.join(base_dir, "cameras.json"), encoding="utf-8") as f:
    base_cams = json.load(f)
with open(os.path.join(base_dir, "landmarks.json"), encoding="utf-8") as f:
    base_marks = json.load(f)
with open(views_path, encoding="utf-8") as f:
    records = json.load(f)

use = np.array([a is not None for a in anchors])
ids = np.nonzero(use)[0]
A = np.array([anchors[i]["position"] for i in ids])
N = np.array([anchors[i]["normal"] for i in ids])
count = len(ids)

# ---------------------------------------------------------------- how reliable is each point?
errors = [[] for _ in range(count)]
for key, cam in base_cams.items():
    p = np.array(cam["projection"])
    seen = np.array(base_marks[f"view_{key}"]["points"])[ids, :2]
    centre = np.array(cam["location"])
    homog = np.hstack([A, np.ones((count, 1))]) @ p.T
    projected = homog[:, :2] / homog[:, 2:]
    to_cam = centre - A
    facing = (N * to_cam).sum(1) / np.linalg.norm(to_cam, axis=1)
    for k in range(count):
        if facing[k] > 0.5:
            errors[k].append(np.linalg.norm(projected[k] - seen[k]))
# The base renders are 1024 px with the eye corners about 230 px apart; express error in that unit.
typical = np.array([np.median(e) if e else 50.0 for e in errors])
reliability = 1.0 / (1.0 + (typical / 6.0) ** 2)
print("POINT_RELIABILITY median px", round(float(np.median(typical)), 1), "points under 0.5:", int((reliability < 0.5).sum()), "of", count)

# Mirror partners on the base head: reflect x and find the nearest point.
mirror = np.array([np.argmin(np.linalg.norm(A - A[k] * np.array([-1, 1, 1]), axis=1)) for k in range(count)])

# ---------------------------------------------------------------- frames
frames = [r for r in records if r["blink"] < 0.35 and r["expression"] < 0.45]
frames = frames[::int(os.environ.get("FRAME_STEP", "2"))]
width, height = records[0].get("size", (1920, 1080))
obs = np.array([np.array(r["points"])[ids, :2] for r in frames])
print("FRAMES used", len(frames), "of", len(records), "yaw range", min(r["yaw"] for r in frames), max(r["yaw"] for r in frames))


def umeyama(src, dst, weights, with_scale=True):
    w = weights / weights.sum()
    mu_s, mu_d = (src * w[:, None]).sum(0), (dst * w[:, None]).sum(0)
    s0, d0 = src - mu_s, dst - mu_d
    cov = (d0 * w[:, None]).T @ s0
    u, sig, vt = np.linalg.svd(cov)
    d = np.eye(3)
    d[2, 2] = np.sign(np.linalg.det(u @ vt))
    r = u @ d @ vt
    scale = (sig * np.diag(d)).sum() / (w * (s0 ** 2).sum(1)).sum() if with_scale else 1.0
    return scale, r, mu_d - scale * r @ mu_s


def solve(focal, iterations=14, verbose=False):
    k_mat = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])
    X = A.copy()
    normals = N.copy()
    poses = [None] * len(frames)
    for it in range(iterations):
        # 1. cameras
        residual = []
        for f in range(len(frames)):
            if poses[f] is None:
                # Start each frame from the tracker's own head angle estimate is unnecessary: solvePnP
                # finds it, starting with the face straight ahead at a plausible distance.
                rvec, tvec = np.zeros(3), np.array([0, 0, focal * 0.15 / 500.0])
                ok, rvec, tvec = cv2.solvePnP(X[reliability > 0.5], obs[f][reliability > 0.5], k_mat, None,
                                              flags=cv2.SOLVEPNP_EPNP)
            else:
                rvec, tvec = poses[f]
            rot = cv2.Rodrigues(rvec)[0]
            cam_pts = X @ rot.T + tvec.reshape(1, 3)
            facing = -(normals @ rot.T * cam_pts).sum(1) / np.linalg.norm(cam_pts, axis=1)
            good = (facing > 0.3) & (reliability > 0.35)
            ok, rvec, tvec = cv2.solvePnP(X[good], obs[f][good], k_mat, None, rvec.copy(), tvec.copy(), True,
                                          cv2.SOLVEPNP_ITERATIVE)
            poses[f] = (rvec.reshape(3), tvec.reshape(3))
            proj, _ = cv2.projectPoints(X[good], rvec, tvec, k_mat, None)
            residual.extend(np.linalg.norm(proj.reshape(-1, 2) - obs[f][good], axis=1))
        # 2. points: weighted linear triangulation with a pull toward the base head's shape
        rots = [cv2.Rodrigues(p[0])[0] for p in poses]
        trans = [p[1] for p in poses]
        projections = [k_mat @ np.hstack([r, t.reshape(3, 1)]) for r, t in zip(rots, trans)]
        new_x = X.copy()
        spread = np.zeros(count)
        view_axis = np.mean([r.T @ np.array([0, 0, 1.0]) for r in rots], axis=0)
        for k in range(count):
            rows, rhs, views = [], [], []
            for f, pm in enumerate(projections):
                cam_pt = rots[f] @ X[k] + trans[f]
                facing = -(rots[f] @ normals[k]) @ cam_pt / np.linalg.norm(cam_pt)
                if facing < 0.4:
                    continue
                u, v = obs[f][k]
                weight = np.sqrt(reliability[k] * facing) / cam_pt[2]  # pixel units
                for row, coord in ((pm[0], u), (pm[1], v)):
                    a = coord * pm[2] - row
                    rows.append(a[:3] * weight)
                    rhs.append(-a[3] * weight)
                views.append(rots[f].T @ np.array([0, 0, 1.0]))
            if len(views) < 2:
                continue
            spread[k] = np.degrees(np.arccos(np.clip(min(np.dot(views[0], v) for v in views), -1, 1)))
            # A pull toward the base head. Across the picture it is gentle, as the images show that
            # directly; along the line of sight, which images taken from one angle can't measure,
            # it can be stronger (PRIOR_DEPTH), so depth stays the base head's unless evidence says
            # otherwise. With frames from many angles both are the same.
            look = view_axis / np.linalg.norm(view_axis)
            along = np.outer(look, look)
            prior_rows = np.sqrt(PRIOR_PLANE) * (np.eye(3) - along) + np.sqrt(PRIOR_DEPTH) * along
            rows.extend(prior_rows)
            rhs.extend(prior_rows @ A[k])
            sol, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
            new_x[k] = sol
        # Remove the free size and placement: express the points in the base head's frame and size.
        s, r, t = umeyama(new_x, A, reliability)
        new_x = (s * new_x @ r.T) + t
        poses = [(cv2.Rodrigues(rot @ r.T)[0].reshape(3), s * tr - rot @ r.T @ t) for rot, tr in zip(rots, trans)]
        # Symmetry, where one side was seen at wider angles than its partner.
        mirrored = new_x[mirror] * np.array([-1, 1, 1])
        share = np.where(spread + spread[mirror] > 0, spread[mirror] / (spread + spread[mirror] + 1e-6), 0.5)
        X = new_x * (1 - 0.5 * share)[:, None] + mirrored * (0.5 * share)[:, None]
        if verbose:
            print(f"  iteration {it}: median reprojection {np.median(residual):.2f} px, 90th {np.percentile(residual, 90):.2f} px")
    # Re-solve cameras once more for the final points.
    final_res = []
    for f in range(len(frames)):
        rvec, tvec = poses[f]
        ok, rvec, tvec = cv2.solvePnP(X[reliability > 0.35], obs[f][reliability > 0.35], k_mat, None,
                                      rvec.reshape(3, 1).copy(), tvec.reshape(3, 1).copy(), True, cv2.SOLVEPNP_ITERATIVE)
        poses[f] = (rvec.reshape(3), tvec.reshape(3))
        proj, _ = cv2.projectPoints(X[reliability > 0.35], rvec, tvec, k_mat, None)
        final_res.extend(np.linalg.norm(proj.reshape(-1, 2) - obs[f][reliability > 0.35], axis=1))
    return X, poses, k_mat, float(np.median(final_res)), float(np.percentile(final_res, 90)), spread


results = {}
for focal in [round(width * f) for f in (0.83, 1.25, 1.9, 2.8, 4.2, 6.3)]:
    X, poses, k_mat, med, p90, spread = solve(focal)
    results[focal] = (X, poses, k_mat, med, p90, spread)
    change = np.linalg.norm(X - A, axis=1) * 1000
    print(f"FOCAL {focal}px: reprojection median {med:.2f} px, 90th {p90:.2f} px; change from base median {np.median(change):.1f} mm, 90th {np.percentile(change, 90):.1f} mm")
best = min(results, key=lambda f: results[f][3])
X, poses, k_mat, med, p90, spread = results[best]
print("BEST_FOCAL", best, "px (", round(best / width * 36, 1), "mm on a full-frame camera )")
_ = solve(best, verbose=True)

out = {
    "ids": ids.tolist(), "points": X.tolist(), "base_points": A.tolist(), "reliability": reliability.tolist(),
    "angle_spread_deg": spread.tolist(), "focal_px": best, "size": [int(width), int(height)],
    "frames": [{"video": r["video"], "frame": r["frame"], "rvec": p[0].tolist(), "tvec": p[1].tolist()} for r, p in zip(frames, poses)],
    "reprojection_px": [med, p90],
}
with open(out_path, "w", encoding="utf-8") as handle:
    json.dump(out, handle)
change = np.linalg.norm(X - A, axis=1) * 1000
print("SHAPE_CHANGE_FROM_BASE_MM median", round(float(np.median(change)), 1), "max", round(float(change.max()), 1))
print("FIT_OK")
