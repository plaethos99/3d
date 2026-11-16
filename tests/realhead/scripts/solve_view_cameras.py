"""Find the camera for chosen video frames against the measured face, for rendering at the same angle.

Usage: solve_view_cameras.py <subject_fit.json> <views.json> <out.json> <video:frame> [...]
Writes, per frame, the camera centre, its rotation (camera to world, Blender axes) and the lens.
"""
import json
import sys

import cv2
import numpy as np

fit_path, views_path, out_path = sys.argv[1:4]
wanted = [w.strip().split(":") for w in sys.argv[4:] if w.strip()]
with open(fit_path, encoding="utf-8") as f:
    fit = json.load(f)
with open(views_path, encoding="utf-8") as f:
    records = json.load(f)
ids = np.array(fit["ids"])
X = np.array(fit["points"])
rel = np.array(fit["reliability"]) > 0.35
focal = fit["focal_px"]
width, height = fit["size"]
k_mat = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1.0]])

out = {}
for video, frame in wanted:
    record = next(r for r in records if r["video"].startswith(video) and r["frame"] == int(frame))
    obs = np.array(record["points"])[ids, :2]
    ok, rvec, tvec = cv2.solvePnP(X[rel], obs[rel], k_mat, None, flags=cv2.SOLVEPNP_EPNP)
    ok, rvec, tvec = cv2.solvePnP(X[rel], obs[rel], k_mat, None, rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE)
    proj, _ = cv2.projectPoints(X[rel], rvec, tvec, k_mat, None)
    error = np.linalg.norm(proj.reshape(-1, 2) - obs[rel], axis=1)
    rot = cv2.Rodrigues(rvec)[0]
    centre = (-rot.T @ tvec).reshape(3)
    # OpenCV cameras look down +Z with +Y down; Blender's look down -Z with +Y up.
    cam_to_world = rot.T @ np.diag([1, -1, -1])
    key = f"{video}_{frame}"
    out[key] = {"video": record["video"], "frame": int(frame), "yaw": record["yaw"], "centre": centre.tolist(),
                "rotation": cam_to_world.tolist(), "lens_mm": focal / width * 36.0, "size": [width, height],
                "projection": (k_mat @ np.hstack([rot, tvec.reshape(3, 1)])).tolist(),
                "reprojection_px": round(float(np.median(error)), 2)}
    print("VIEW", key, "yaw", record["yaw"], "reprojection median px", out[key]["reprojection_px"],
          "camera distance m", round(float(np.linalg.norm(centre)), 3))
with open(out_path, "w", encoding="utf-8") as handle:
    json.dump(out, handle)
print("CAMERAS_OK")
