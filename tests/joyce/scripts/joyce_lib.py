"""Geometry, texture and image helpers for the agent-driven Joyce build.

Every mesh here is constructed by explicit modeling steps (lofted sections,
sculpt offsets, placed locks). No generative mesh model is involved.
"""
import math
import os

import bpy
import numpy as np


# ---------------------------------------------------------------- interpolation
def catmull(values, count):
    """Resample a sequence of arrays smoothly to `count` samples."""
    values = np.asarray(values, dtype=float)
    n = len(values)
    if n == 1:
        return np.repeat(values, count, axis=0)
    out = []
    for q in np.linspace(0.0, n - 1.0, count):
        i = min(int(q), n - 2)
        t = q - i
        p0 = values[max(i - 1, 0)]
        p1 = values[i]
        p2 = values[i + 1]
        p3 = values[min(i + 2, n - 1)]
        out.append(
            0.5
            * (
                2 * p1
                + (p2 - p0) * t
                + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t
                + (3 * p1 - p0 - 3 * p2 + p3) * t * t * t
            )
        )
    return np.array(out)


def normalized(v, axis=-1):
    length = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(length, 1e-9)


# ---------------------------------------------------------------- mesh creation
def grid_faces(rows, cols, cyclic):
    faces = []
    span = cols if cyclic else cols - 1
    for i in range(rows - 1):
        for j in range(span):
            a = i * cols + j
            b = i * cols + (j + 1) % cols
            c = (i + 1) * cols + (j + 1) % cols
            d = (i + 1) * cols + j
            faces.append((a, b, c, d))
    return faces


def make_object(name, verts, faces, loop_uvs=None, material=None, smooth=True, collection=None):
    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata([tuple(v) for v in np.asarray(verts).tolist()], [], faces)
    mesh.update()
    if loop_uvs is not None:
        layer = mesh.uv_layers.new(name="UVMap")
        layer.uv.foreach_set("vector", np.asarray(loop_uvs, dtype=np.float32).ravel())
    if smooth:
        mesh.shade_smooth()
    obj = bpy.data.objects.new(name, mesh)
    (collection or bpy.context.scene.collection).objects.link(obj)
    if material is not None:
        mesh.materials.append(material)
    return obj


def grid_object(name, rings, material=None, cyclic=True, cap_start=False, cap_end=False,
                uv_u=None, uv_v=None, smooth=True, flip=False):
    """Create a quad mesh from an array of rings shaped (rows, cols, 3)."""
    rings = np.asarray(rings, dtype=float)
    rows, cols, _ = rings.shape
    verts = rings.reshape(-1, 3).tolist()
    faces = grid_faces(rows, cols, cyclic)
    span = cols if cyclic else cols - 1

    uv_u = np.linspace(0, 1, span + 1) if uv_u is None else np.asarray(uv_u, dtype=float)
    uv_v = np.linspace(0, 1, rows) if uv_v is None else np.asarray(uv_v, dtype=float)
    loop_uvs = []
    for i in range(rows - 1):
        for j in range(span):
            loop_uvs += [
                (uv_u[j], uv_v[i]),
                (uv_u[j + 1], uv_v[i]),
                (uv_u[j + 1], uv_v[i + 1]),
                (uv_u[j], uv_v[i + 1]),
            ]

    def add_cap(row, reverse):
        index = len(verts)
        verts.append(rings[row].mean(axis=0).tolist())
        for j in range(span):
            a = row * cols + j
            b = row * cols + (j + 1) % cols
            faces.append((a, b, index) if not reverse else (b, a, index))
            loop_uvs.extend([(uv_u[j], uv_v[row])] * 3)

    if cap_start:
        add_cap(0, True)
    if cap_end:
        add_cap(rows - 1, False)

    if flip:
        faces = [tuple(reversed(f)) for f in faces]
        rebuilt = []
        cursor = 0
        for f in faces:
            rebuilt.extend(reversed(loop_uvs[cursor:cursor + len(f)]))
            cursor += len(f)
        loop_uvs = rebuilt

    return make_object(name, verts, faces, loop_uvs, material, smooth)


def ring_angles(cols):
    return np.linspace(0.0, 2 * math.pi, cols, endpoint=False)


def super_ring(theta, power):
    c, s = np.cos(theta), np.sin(theta)
    e = 2.0 / power
    return np.sign(c) * np.abs(c) ** e, np.sign(s) * np.abs(s) ** e


def tube_rings(path, radii, rows, cols, side=(1.0, 0.0, 0.0), power=2.0):
    """Sweep ellipses along a path. `radii` holds (side radius, other radius)."""
    centers = catmull(path, rows)
    rad = catmull(radii, rows)
    tangents = np.gradient(centers, axis=0)
    tangents = normalized(tangents)

    side = np.array(side, dtype=float)
    theta = ring_angles(cols)
    cx, cy = super_ring(theta, power)
    rings = np.zeros((rows, cols, 3))
    current = side - np.dot(side, tangents[0]) * tangents[0]
    current = normalized(current)
    for i in range(rows):
        current = current - np.dot(current, tangents[i]) * tangents[i]
        current = normalized(current)
        binormal = np.cross(tangents[i], current)
        rings[i] = (
            centers[i]
            + np.outer(cx * rad[i, 0], current)
            + np.outer(cy * rad[i, 1], binormal)
        )
    return rings


def tube(name, path, radii, material=None, rows=24, cols=20, side=(1, 0, 0), power=2.0,
         cap_start=True, cap_end=True, uv_scale=None):
    rings = tube_rings(path, radii, rows, cols, side, power)
    uv_u = uv_v = None
    if uv_scale is not None:
        mean_radius = float(np.mean(radii))
        circumference = 2 * math.pi * mean_radius
        lengths = np.linalg.norm(np.diff(rings.mean(axis=1), axis=0), axis=1)
        arc = np.concatenate([[0.0], np.cumsum(lengths)])
        uv_u = np.linspace(0, circumference * uv_scale[0], cols + 1)
        uv_v = arc * uv_scale[1] + uv_scale[2] if len(uv_scale) > 2 else arc * uv_scale[1]
    return grid_object(name, rings, material, True, cap_start, cap_end, uv_u, uv_v)


def ellipsoid_rings(center, radii, rows, cols):
    polar = np.linspace(0.02, math.pi - 0.02, rows)
    theta = ring_angles(cols)
    rings = np.zeros((rows, cols, 3))
    for i, p in enumerate(polar):
        rings[i, :, 0] = center[0] + radii[0] * np.sin(p) * np.sin(theta)
        rings[i, :, 1] = center[1] - radii[1] * np.sin(p) * np.cos(theta)
        rings[i, :, 2] = center[2] + radii[2] * np.cos(p)
    return rings


def ellipsoid(name, center, radii, material=None, rows=16, cols=24):
    rings = ellipsoid_rings(center, radii, rows, cols)
    return grid_object(name, rings, material, True, True, True)


def mirror_x(point):
    return (-point[0], point[1], point[2])


# ---------------------------------------------------------------- materials
def new_material(name, color=(0.5, 0.5, 0.5, 1.0), roughness=0.6):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def srgb(hex_value):
    """Convert '#rrggbb' to a linear RGBA colour for shader inputs."""
    hex_value = hex_value.lstrip("#")
    out = []
    for i in (0, 2, 4):
        c = int(hex_value[i:i + 2], 16) / 255.0
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return (out[0], out[1], out[2], 1.0)


def attach_color_image(mat, image):
    nodes = mat.node_tree.nodes
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    mat.node_tree.links.new(tex.outputs["Color"], nodes["Principled BSDF"].inputs["Base Color"])
    return tex


def attach_bump_image(mat, image, strength=0.5, distance=0.003):
    nodes = mat.node_tree.nodes
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = image
    image.colorspace_settings.name = "Non-Color"
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = strength
    bump.inputs["Distance"].default_value = distance
    mat.node_tree.links.new(tex.outputs["Color"], bump.inputs["Height"])
    mat.node_tree.links.new(bump.outputs["Normal"], nodes["Principled BSDF"].inputs["Normal"])


def array_to_image(name, array, folder=None):
    """Save an array shaped (H, W, 3|4) with values 0..1, first row at the bottom."""
    height, width = array.shape[:2]
    if array.shape[2] == 3:
        array = np.dstack([array, np.ones((height, width))])
    image = bpy.data.images.new(name, width, height, alpha=True)
    image.pixels.foreach_set(np.clip(array, 0, 1).astype(np.float32).ravel())
    if folder:
        os.makedirs(folder, exist_ok=True)
        image.filepath_raw = os.path.join(folder, name + ".png")
        image.file_format = "PNG"
        image.save()
    image.pack()
    return image


# ---------------------------------------------------------------- image comparison
def load_rgba(path):
    image = bpy.data.images.load(path, check_existing=False)
    width, height = image.size
    data = np.empty(width * height * 4, dtype=np.float32)
    image.pixels.foreach_get(data)
    bpy.data.images.remove(image)
    return data.reshape(height, width, 4)


def save_rgba(path, array):
    height, width = array.shape[:2]
    image = bpy.data.images.new("tmp_save", width, height, alpha=True)
    image.pixels.foreach_set(np.clip(array, 0, 1).astype(np.float32).ravel())
    image.filepath_raw = path
    image.file_format = "PNG"
    image.save()
    bpy.data.images.remove(image)


def resize_nearest(array, height, width):
    rows = (np.arange(height) * array.shape[0] / height).astype(int)
    cols = (np.arange(width) * array.shape[1] / width).astype(int)
    return array[rows][:, cols]


def silhouette_iou(render_rgba, reference_rgba):
    """Return intersection-over-union of the two alpha silhouettes."""
    ref = resize_nearest(reference_rgba, *render_rgba.shape[:2])
    a = render_rgba[..., 3] > 0.5
    b = ref[..., 3] > 0.5
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / max(union, 1)), a, b


def over_background(rgba, background=(0.86, 0.85, 0.82)):
    alpha = rgba[..., 3:4]
    rgb = rgba[..., :3] * alpha + np.array(background) * (1 - alpha)
    return np.dstack([rgb, np.ones(alpha.shape)])


def comparison_sheet(render_rgba, reference_rgba):
    """Reference, render and a silhouette difference, side by side."""
    height, width = render_rgba.shape[:2]
    ref = resize_nearest(reference_rgba, height, width)
    a = render_rgba[..., 3] > 0.5
    b = ref[..., 3] > 0.5
    diff = np.ones((height, width, 4))
    diff[..., :3] = 0.92
    diff[a & b, :3] = (0.62, 0.62, 0.62)
    diff[a & ~b, :3] = (0.85, 0.20, 0.15)   # model only
    diff[~a & b, :3] = (0.15, 0.35, 0.85)   # reference only
    return np.hstack([over_background(ref), over_background(render_rgba), diff])
