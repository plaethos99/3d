"""Find the size and placement of a fitted head on an existing body, from a shared reference image.

Usage: place_head.py <subject_fit.json> <views.json> <view_name> <camera.json> <anchor_x,y,z> <out.json>

camera.json describes the camera that renders the body like that reference: location, aim point,
lens (mm), vertical sensor (mm) and image size. For each candidate size the head's face points are
placed by perspective-n-point so they land on the reference's tracked points. In one picture size
and distance trade off, so the size chosen is the one that puts the eyes' midpoint closest to
anchor (where the body's own head had its eyes).
"""
import json
import sys

import cv2
import numpy as np

fit_path, views_path, view_name, cam_path, anchor, out = sys.argv[1:7]
anchor = np.array([float(v) for v in anchor.split(",")])
with open(fit_path, encoding="utf-8") as f:
    fit = json.load(f)
with open(views_path, encoding="utf-8") as f:
    record = next(r for r in json.load(f) if r["video"] == view_name)
with open(cam_path, encoding="utf-8") as f:
    cam = json.load(f)

ids = np.array(fit["ids"])
X = np.array(fit["points"])
reliable = np.array(fit["reliability"]) > 0.35
uv = np.array(record["points"])[ids, :2]
w, h = cam["size"]
focal = cam["lens_mm"] / cam["sensor_mm"] * h
K = np.array([[focal, 0, w / 2], [0, focal, h / 2], [0, 0, 1.0]])

# Body camera: OpenCV axes (x right, y down, z forward) from location and aim.
location, aim = np.array(cam["location"]), np.array(cam["aim"])
forward = (aim - location) / np.linalg.norm(aim - location)
right = np.cross(forward, np.array((0, 0, 1.0)))
right /= np.linalg.norm(right)
down = np.cross(forward, right)
R_bc = np.stack([right, down, forward])  # world -> camera rotation
t_bc = -R_bc @ location

rows = []
for scale in np.arange(0.9, 1.81, 0.02):
    ok, rvec, tvec = cv2.solvePnP(X[reliable] * scale, uv[reliable], K, None, flags=cv2.SOLVEPNP_EPNP)
    ok, rvec, tvec = cv2.solvePnP(X[reliable] * scale, uv[reliable], K, None, rvec, tvec, True, cv2.SOLVEPNP_ITERATIVE)
    R_hc = cv2.Rodrigues(rvec)[0]
    proj, _ = cv2.projectPoints(X[reliable] * scale, rvec, tvec, K, None)
    err = float(np.median(np.linalg.norm(proj.reshape(-1, 2) - uv[reliable], axis=1)))
    R_hb = R_bc.T @ R_hc
    T = R_bc.T @ (tvec.reshape(3) - t_bc)
    eyes_mid = T  # the head's origin is the midpoint between its eyes
    rows.append((float(np.linalg.norm(eyes_mid - anchor)), scale, err, R_hb, T))
best = min(rows, key=lambda r: r[0])
gap, scale, err, R_hb, T = best
yaw = np.degrees(np.arctan2(R_hb[1, 0], R_hb[0, 0]))
pitch = np.degrees(np.arcsin(np.clip(R_hb[2, 1], -1, 1)))
roll = np.degrees(np.arctan2(-R_hb[2, 0], R_hb[2, 2]))
print("PLACE scale", round(float(scale), 3), "reprojection px", round(err, 2), "eye midpoint", np.round(T, 3).tolist(),
      "distance to anchor mm", round(gap * 1000, 1), "turn deg (yaw, nod, tilt)", round(float(yaw), 1), round(float(pitch), 1), round(float(roll), 1))
for g, s, e, _, t in rows[::5]:
    print(f"  size {s:.2f}: reprojection {e:.2f} px, eyes {np.round(t, 3).tolist()}, {g * 1000:.0f} mm from anchor")
with open(out, "w", encoding="utf-8") as handle:
    json.dump({"scale": float(scale), "rotation": R_hb.tolist(), "translation": T.tolist(), "reprojection_px": err,
               "turn_deg": [float(yaw), float(pitch), float(roll)]}, handle, indent=1)
print("PLACE_OK")
