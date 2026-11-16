"""Face-specific checks, because a body silhouette score cannot judge a face.

Three measures, all at the reference image's 832 x 1248 scale:
- landmark error: where facial features land in the image against where they sit in
  the reference (reference positions were measured from a gridded close-up)
- head outline match: silhouette overlap for the head and hair only
- head structure: local image structure (SSIM) inside the head region
"""
import json

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view

# Measured from renders/face/ref_stand_zoom.png. Valid for refs/stand.png only.
REFERENCE_STAND = {
    "eye_near": (452.7, 328.3),
    "eye_far": (514.0, 331.7),
    "nose_tip": (493.3, 361.7),
    "mouth_near": (448.3, 384.0),
    "mouth_far": (511.7, 390.0),
    "mouth_center": (483.3, 392.7),
    "chin": (490.0, 436.0),
    "ear_near": (340.0, 366.7),
}
HEAD_BOX = (150, 450)  # rows from the top, at 1248 height


def _evaluated_coords(obj, depsgraph):
    evaluated = obj.evaluated_get(depsgraph)
    mesh = evaluated.to_mesh()
    coords = np.empty(len(mesh.vertices) * 3)
    mesh.vertices.foreach_get("co", coords)
    evaluated.to_mesh_clear()
    return coords.reshape(-1, 3)


def _rest_coords(obj):
    coords = np.empty(len(obj.data.vertices) * 3)
    obj.data.vertices.foreach_get("co", coords)
    return coords.reshape(-1, 3)


def model_landmarks(spec):
    """Find each feature on the posed mesh. `spec` gives head-local x and z values."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    head = bpy.data.objects["head"]
    rest = _rest_coords(head)
    posed = _evaluated_coords(head, depsgraph)
    center_z = spec["head_center_z"]

    def on_face(x, z):
        front = rest[:, 1] < rest[:, 1].mean()
        distance = np.hypot(rest[:, 0] - x, rest[:, 2] - (center_z + z)) + (~front) * 10
        return posed[int(np.argmin(distance))]

    def front_of_eye(name):
        rest_eye = _rest_coords(bpy.data.objects[name])
        posed_eye = _evaluated_coords(bpy.data.objects[name], depsgraph)
        return posed_eye[int(np.argmin(rest_eye[:, 1]))]

    corner = spec["mouth_corner"]
    points = {
        "eye_near": front_of_eye("eye_R"),
        "eye_far": front_of_eye("eye_L"),
        "nose_tip": posed[int(np.argmin(np.where(np.abs(rest[:, 0]) < 0.004, rest[:, 1], 9)))],
        "mouth_near": on_face(-corner[0], corner[1]),
        "mouth_far": on_face(corner[0], corner[1]),
        "mouth_center": on_face(0.0, spec["mouth_center_z"]),
        "chin": posed[int(np.argmin(np.where((np.abs(rest[:, 0]) < 0.01) & (rest[:, 1] < -0.09), rest[:, 2], 9)))],
        "ear_near": _evaluated_coords(bpy.data.objects["ear_R"], depsgraph).mean(axis=0),
    }
    return points


def landmark_report(scene, camera, spec):
    from mathutils import Vector

    errors = {}
    for name, world in model_landmarks(spec).items():
        co = world_to_camera_view(scene, camera, Vector(tuple(world)))
        px = (co.x * 832.0, (1.0 - co.y) * 1248.0)
        ref = REFERENCE_STAND[name]
        errors[name] = round(float(np.hypot(px[0] - ref[0], px[1] - ref[1])), 1)
    mean = round(float(np.mean(list(errors.values()))), 1)
    return mean, errors


def _grey(rgba, background=0.86):
    alpha = rgba[..., 3]
    rgb = rgba[..., :3] * alpha[..., None] + background * (1 - alpha[..., None])
    return rgb @ np.array((0.2126, 0.7152, 0.0722))


def head_scores(render, reference):
    """Return head silhouette overlap and block SSIM over the head region."""
    height, width = render.shape[:2]
    rows = (np.arange(height) * reference.shape[0] / height).astype(int)
    cols = (np.arange(width) * reference.shape[1] / width).astype(int)
    ref = reference[rows][:, cols]
    # Stored rows start at the bottom of the picture.
    top = int(height - HEAD_BOX[1] * height / 1248)
    bottom = int(height - HEAD_BOX[0] * height / 1248)
    a = render[top:bottom]
    b = ref[top:bottom]
    mask_a, mask_b = a[..., 3] > 0.5, b[..., 3] > 0.5
    iou = float((mask_a & mask_b).sum() / max((mask_a | mask_b).sum(), 1))

    grey_a, grey_b = _grey(a), _grey(b)
    block = max(int(8 * height / 1248), 6)
    h, w = grey_a.shape[0] // block, grey_a.shape[1] // block
    grey_a = grey_a[: h * block, : w * block].reshape(h, block, w, block)
    grey_b = grey_b[: h * block, : w * block].reshape(h, block, w, block)
    mean_a, mean_b = grey_a.mean(axis=(1, 3)), grey_b.mean(axis=(1, 3))
    var_a, var_b = grey_a.var(axis=(1, 3)), grey_b.var(axis=(1, 3))
    cov = ((grey_a - mean_a[:, None, :, None]) * (grey_b - mean_b[:, None, :, None])).mean(axis=(1, 3))
    c1, c2 = 0.01**2, 0.03**2
    ssim = ((2 * mean_a * mean_b + c1) * (2 * cov + c2)) / (
        (mean_a**2 + mean_b**2 + c1) * (var_a + var_b + c2)
    )
    return round(iou, 4), round(float(ssim.mean()), 4)


def load_spec(path):
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)
