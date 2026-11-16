"""Grow short, tightly curled hair on the scalp, where the hair mask says there is hair.

Usage: blender -b head.blend -P add_hair_strands.py -- <hair_mask.png> <out.blend>

Strands come out of the scalp and coil, like short afro-textured hair. Their density and length
follow the hair mask, so they thin out at the hairline. The scalp underneath is pulled in by about
the height the coils add, so the outline fitted to the photos stays where it was.
"""
import os
import sys

import bpy
import numpy as np

args = sys.argv[sys.argv.index("--") + 1:]
mask_path, out = args[0], args[1]
LENGTH = float(os.environ.get("HAIR_LENGTH", "0.005"))
TAPER_STEPS = int(os.environ.get("HAIR_TAPER_STEPS", "6"))
PULL_IN = float(os.environ.get("HAIR_PULL_IN", "0.0045"))
PARENTS = int(os.environ.get("HAIR_PARENTS", "40000"))
CHILDREN = int(os.environ.get("HAIR_CHILDREN", "6"))

head = bpy.data.objects["head"]
mesh = head.data
image = bpy.data.images.load(mask_path)
w, h = image.size
pixels = np.empty(w * h * 4, dtype=np.float32)
image.pixels.foreach_get(pixels)
mask = pixels.reshape(h, w, 4)[:, :, 0]  # Blender's rows run bottom to top, like UV v

# Each vertex takes the hair amount at its place in the texture, averaged over its corners.
uv = np.empty(len(mesh.loops) * 2)
mesh.uv_layers.active.data.foreach_get("uv", uv)
uv = uv.reshape(-1, 2)
loop_vertex = np.empty(len(mesh.loops), dtype=np.int64)
mesh.loops.foreach_get("vertex_index", loop_vertex)
cols = np.clip((uv[:, 0] * w).astype(int), 0, w - 1)
rows = np.clip((uv[:, 1] * h).astype(int), 0, h - 1)
values = mask[rows, cols]
total = np.bincount(loop_vertex, weights=values, minlength=len(mesh.vertices))
count = np.bincount(loop_vertex, minlength=len(mesh.vertices))
weight = total / np.maximum(count, 1)
group = head.vertex_groups.get("hair") or head.vertex_groups.new(name="hair")
for index, value in enumerate(weight):
    if value > 0.02:
        group.add([index], float(value), "REPLACE")
print("HAIR_VERTICES", int((weight > 0.5).sum()), "partly", int(((weight > 0.02) & (weight <= 0.5)).sum()), "of", len(weight), "mask mean", round(float(mask.mean()), 3), "uv range", uv.min(0).round(2), uv.max(0).round(2))

# Real hair gets shorter toward the hairline instead of starting at full length. Spread the hair
# amount over the surface a few times, so it fades with distance from the edge, and use that
# faded amount for strand length (and for how far the scalp is pulled in under it).
edges = np.empty(len(mesh.edges) * 2, dtype=np.int64)
mesh.edges.foreach_get("vertices", edges)
edges = edges.reshape(-1, 2)
spread = weight.copy()
for _ in range(TAPER_STEPS):
    total = np.bincount(edges[:, 0], weights=spread[edges[:, 1]], minlength=len(spread)) +         np.bincount(edges[:, 1], weights=spread[edges[:, 0]], minlength=len(spread))
    degree = np.bincount(edges.ravel(), minlength=len(spread))
    spread = 0.5 * spread + 0.5 * total / np.maximum(degree, 1)
length_weight = np.clip((spread - 0.35) / 0.5, 0, 1) * weight
length_weight = length_weight * length_weight * (3 - 2 * length_weight)
length_group = head.vertex_groups.get("hair_length") or head.vertex_groups.new(name="hair_length")
for index, value in enumerate(length_weight):
    if value > 0.02:
        length_group.add([index], float(value), "REPLACE")
print("FULL_LENGTH_VERTICES", int((length_weight > 0.9).sum()))

# Pull the scalp in under the coils.
coords = np.empty(len(mesh.vertices) * 3)
mesh.vertices.foreach_get("co", coords)
normals = np.empty(len(mesh.vertices) * 3)
mesh.vertices.foreach_get("normal", normals)
coords = coords.reshape(-1, 3) - normals.reshape(-1, 3) * (length_weight * PULL_IN)[:, None]
mesh.vertices.foreach_set("co", coords.ravel())
mesh.update()

# Hair material: real hair's colour comes from melanin; very dark brown-black here.
mat = bpy.data.materials.new("hair_strands")
mat.use_nodes = True
nodes, links = mat.node_tree.nodes, mat.node_tree.links
nodes.remove(nodes["Principled BSDF"])
hair_bsdf = nodes.new("ShaderNodeBsdfHairPrincipled")
hair_bsdf.parametrization = "MELANIN"
hair_bsdf.inputs["Melanin"].default_value = 0.96
hair_bsdf.inputs["Melanin Redness"].default_value = 0.45
hair_bsdf.inputs["Roughness"].default_value = 0.45
hair_bsdf.inputs["Radial Roughness"].default_value = 0.6
hair_bsdf.inputs["Coat"].default_value = 0.05
links.new(hair_bsdf.outputs[0], nodes["Material Output"].inputs["Surface"])
mesh.materials.append(mat)
slot = len(mesh.materials)

bpy.context.view_layer.objects.active = head
modifier = head.modifiers.new("hair", "PARTICLE_SYSTEM")
system = modifier.particle_system
settings = system.settings
settings.type = "HAIR"
settings.count = PARENTS
settings.hair_length = LENGTH
settings.emit_from = "FACE"
settings.use_emit_random = True
settings.material = slot
settings.child_type = "INTERPOLATED"
settings.rendered_child_count = CHILDREN
if hasattr(settings, "child_percent"):
    settings.child_percent = 2
elif hasattr(settings, "child_nbr"):
    settings.child_nbr = 2
settings.child_length = 1.0
settings.kink = "CURL"
settings.kink_amplitude = 0.0012
settings.kink_frequency = 5.0
settings.kink_shape = 0.0
settings.roughness_1 = 0.0012
settings.roughness_1_size = 0.4
settings.roughness_endpoint = 0.001
# Each coil a little different, so the hair doesn't form a visible repeating weave.
settings.roughness_2 = 0.0015
settings.roughness_2_size = 1.0
settings.kink_amplitude_random = 0.6
settings.clump_factor = 0.35
settings.radius_scale = 0.00012
settings.root_radius = 1.0
settings.tip_radius = 0.5
settings.use_close_tip = True
settings.display_step = 3
settings.render_step = 4
system.vertex_group_density = "hair"
system.vertex_group_length = "hair_length"
bpy.ops.wm.save_as_mainfile(filepath=out)
print("STRANDS_OK", PARENTS, "roots,", PARENTS * CHILDREN, "rendered strands, length", LENGTH)
