"""Bake what the texture projection needs from Blender, as numpy files.

Usage: blender -b head.blend -P bake_maps.py -- <view_cameras.json> <out_dir> <texture_size> <view_scale>
Writes:
  uv_position.npy, uv_normal.npy  where each texel of the head's texture sits in 3D, and its normal
  seen_<key>.npy                  for each camera, which texels it can see (1) or not (0)
"""
import json
import os
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector

sys.path.append(os.path.dirname(__file__))
import headcam  # noqa: E402

args = sys.argv[sys.argv.index("--") + 1:]
cams_path, out_dir, size, view_scale = args[0], args[1], int(args[2]), float(args[3])
os.makedirs(out_dir, exist_ok=True)
with open(cams_path, encoding="utf-8") as f:
    views = json.load(f)

scene = bpy.context.scene
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 1
scene.cycles.use_denoising = False
scene.view_settings.view_transform = "Standard"
head = bpy.data.objects["head"]
for obj in bpy.data.objects:
    if obj.type == "MESH" and obj.name != "head":
        obj.hide_render = True
for poly in head.data.polygons:
    poly.use_smooth = True


def emission_material(source, offset=0.0):
    mat = bpy.data.materials.new(f"emit_{source}")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    geo = nodes.new("ShaderNodeNewGeometry")
    emit = nodes.new("ShaderNodeEmission")
    out = nodes.new("ShaderNodeOutputMaterial")
    shift = nodes.new("ShaderNodeVectorMath")
    shift.operation = "ADD"
    shift.inputs[1].default_value = (offset, offset, offset)
    links.new(geo.outputs[source], shift.inputs[0])
    links.new(shift.outputs[0], emit.inputs["Color"])
    links.new(emit.outputs[0], out.inputs["Surface"])
    target = nodes.new("ShaderNodeTexImage")
    nodes.active = target
    return mat, target


def bake(source):
    mat, target = emission_material(source)
    image = bpy.data.images.new(f"bake_{source}", size, size, alpha=False, float_buffer=True)
    image.colorspace_settings.name = "Non-Color"
    target.image = image
    head.data.materials.clear()
    head.data.materials.append(mat)
    for other in bpy.context.view_layer.objects:
        other.select_set(False)
    bpy.context.view_layer.objects.active = head
    head.select_set(True)
    scene.render.bake.margin = 4
    bpy.ops.object.bake(type="EMIT")
    pixels = np.empty(size * size * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    # Blender stores rows bottom to top; flip so row 0 is the top, as images are usually read.
    return pixels.reshape(size, size, 4)[::-1, :, :3].copy()


if True:
    position = bake("Position")
    normal = bake("Normal")
    covered = np.linalg.norm(normal, axis=2) > 0.5
    np.save(os.path.join(out_dir, "uv_position.npy"), position)
    np.save(os.path.join(out_dir, "uv_normal.npy"), normal)
    np.save(os.path.join(out_dir, "uv_covered.npy"), covered)
    print("BAKED", size, "texels on the head", int(covered.sum()), f"({covered.mean() * 100:.0f}% of the texture)")

# What each camera can see. Every vertex is traced toward each camera; one that hits the head on
# the way is hidden. That is stored on the mesh and baked into a texture, one per camera.
from mathutils.bvhtree import BVHTree  # noqa: E402

tree = BVHTree.FromObject(head, bpy.context.evaluated_depsgraph_get())
mesh = head.data
count = len(mesh.vertices)
coords = np.empty(count * 3)
mesh.vertices.foreach_get("co", coords)
coords = coords.reshape(-1, 3)
normals = np.empty(count * 3)
mesh.vertices.foreach_get("normal", normals)
normals = normals.reshape(-1, 3)
for key, view in views.items():
    centre = np.array(view["centre"])
    seen = np.zeros(count, dtype=np.float32)
    for k in range(count):
        to_cam = centre - coords[k]
        distance = float(np.linalg.norm(to_cam))
        direction = Vector(to_cam / distance)
        if direction.dot(Vector(normals[k])) <= 0.0:
            continue
        start = Vector(coords[k] + normals[k] * 0.0005)
        hit = tree.ray_cast(start, direction, distance)
        seen[k] = 0.0 if hit[0] is not None else 1.0
    name = f"seen_{key}"
    attribute = mesh.color_attributes.new(name, "FLOAT_COLOR", "POINT")
    attribute.data.foreach_set("color", np.repeat(seen, 4))
    mat = bpy.data.materials.new(f"emit_{name}")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    attr = nodes.new("ShaderNodeVertexColor")
    attr.layer_name = name
    emit = nodes.new("ShaderNodeEmission")
    out = nodes.new("ShaderNodeOutputMaterial")
    links.new(attr.outputs["Color"], emit.inputs["Color"])
    links.new(emit.outputs[0], out.inputs["Surface"])
    target = nodes.new("ShaderNodeTexImage")
    image = bpy.data.images.new(f"bake_{name}", size, size, alpha=False, float_buffer=True)
    image.colorspace_settings.name = "Non-Color"
    target.image = image
    nodes.active = target
    head.data.materials.clear()
    head.data.materials.append(mat)
    bpy.ops.object.bake(type="EMIT")
    pixels = np.empty(size * size * 4, dtype=np.float32)
    image.pixels.foreach_get(pixels)
    np.save(os.path.join(out_dir, f"seen_{key}.npy"), pixels.reshape(size, size, 4)[::-1, :, 0].astype(np.float16))
    print("SEEN", key, f"{seen.mean() * 100:.0f}% of vertices visible", flush=True)
print("BAKE_OK")
