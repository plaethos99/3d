"""Replace the base eyes with eyeballs sized and placed from the measured eyelids.

Usage: blender -b head.blend -P add_eyes.py -- <subject_fit.json> <out.blend> [iris_rgb]
Eyeballs are 24 mm across, as adult eyes are. Each sits where the base head's eye sits relative
to its eye corners, scaled to the subject's eye opening.
The iris, pupil and the wet shine of the cornea are painted in the eye's own material.
"""
import json
import os
import sys

import bpy
import numpy as np
from mathutils import Vector

args = sys.argv[sys.argv.index("--") + 1:]
fit_path, out = args[0], args[1]
iris_rgb = tuple(float(v) for v in args[2].split(",")) if len(args) > 2 else (0.009, 0.0045, 0.0025)
WHITE = tuple(float(v) for v in os.environ.get("SCLERA_RGB", "0.34,0.29,0.26").split(","))
with open(fit_path, encoding="utf-8") as f:
    fit = json.load(f)
index = {i: k for k, i in enumerate(fit["ids"])}
X = np.array(fit["points"])

# A base head that brings its own eyeballs (as the stylized one does) had its sockets sculpted
# around them, and the reshape moved them along with the face. Their place and size are kept.
base_eyes = {}
for side in "LR":
    obj = bpy.data.objects.get(f"eye.{side}")
    if obj is not None:
        co = np.array([v.co for v in obj.data.vertices])
        base_eyes[side] = (co.mean(axis=0), float(np.ptp(co[:, 0]) / 2))
for name in ("sclera.L", "sclera.R", "iris.L", "iris.R", "eye.L", "eye.R"):
    if name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)

# Adult human eyes are 24 mm across. Cartoon characters often have far larger ones; EYE_RADIUS,
# IRIS_SHARE and BASE_EYE_X set those for other heads.
RADIUS = float(os.environ.get("EYE_RADIUS", "0.012"))
IRIS_RADIUS = RADIUS * float(os.environ.get("IRIS_SHARE", "0.525"))
PUPIL_RADIUS = IRIS_RADIUS * float(os.environ.get("PUPIL_SHARE", "0.33"))
SPAN = RADIUS / 0.6  # the colour ramp below covers 0..SPAN of distance from the eye's axis


def eye_material(name):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    coords = nodes.new("ShaderNodeTexCoord")
    sep = nodes.new("ShaderNodeSeparateXYZ")
    links.new(coords.outputs["Object"], sep.inputs[0])
    # Distance from the eye's forward axis (object -Y), in metres.
    radial = nodes.new("ShaderNodeMath")
    radial.operation = "SQRT"
    sx, sz = nodes.new("ShaderNodeMath"), nodes.new("ShaderNodeMath")
    sx.operation = sz.operation = "MULTIPLY"
    links.new(sep.outputs["X"], sx.inputs[0])
    links.new(sep.outputs["X"], sx.inputs[1])
    links.new(sep.outputs["Z"], sz.inputs[0])
    links.new(sep.outputs["Z"], sz.inputs[1])
    add = nodes.new("ShaderNodeMath")
    links.new(sx.outputs[0], add.inputs[0])
    links.new(sz.outputs[0], add.inputs[1])
    links.new(add.outputs[0], radial.inputs[0])
    front = nodes.new("ShaderNodeMath")
    front.operation = "LESS_THAN"
    links.new(sep.outputs["Y"], front.inputs[0])
    front.inputs[1].default_value = 0.0
    ramp = nodes.new("ShaderNodeValToRGB")
    elements = ramp.color_ramp.elements
    elements[0].position, elements[0].color = 0.0, (0.01, 0.01, 0.01, 1)
    elements[1].position, elements[1].color = PUPIL_RADIUS / SPAN + 0.004, (*iris_rgb, 1)
    mid = elements.new(IRIS_RADIUS / SPAN - 0.012)
    mid.color = (iris_rgb[0] * 1.6, iris_rgb[1] * 1.6, iris_rgb[2] * 1.6, 1)
    rim = elements.new(IRIS_RADIUS / SPAN)
    rim.color = (0.02, 0.012, 0.008, 1)
    white = elements.new(IRIS_RADIUS / SPAN + 0.02)
    white.color = (*WHITE, 1)
    edge = elements.new(0.6)
    edge.color = (0.36, 0.27, 0.24, 1)
    scale = nodes.new("ShaderNodeMath")
    scale.operation = "MULTIPLY"
    links.new(radial.outputs[0], scale.inputs[0])
    scale.inputs[1].default_value = 1 / SPAN
    # Behind the eye's equator it is all white.
    back = nodes.new("ShaderNodeMath")
    back.operation = "MAXIMUM"
    links.new(scale.outputs[0], back.inputs[0])
    flip = nodes.new("ShaderNodeMath")
    flip.operation = "SUBTRACT"
    flip.inputs[0].default_value = 1.0
    links.new(front.outputs[0], flip.inputs[1])
    links.new(flip.outputs[0], back.inputs[1])
    links.new(back.outputs[0], ramp.inputs[0])
    links.new(ramp.outputs[0], bsdf.inputs["Base Color"])
    # A wet, glossy cornea reflects whatever surrounds it. Under the soft, even light used to match
    # the photos that reflection would be a grey veil, so the shine is kept to a tight highlight.
    bsdf.inputs["Roughness"].default_value = 0.08
    bsdf.inputs["Specular IOR Level"].default_value = 0.25
    bsdf.inputs["Coat Weight"].default_value = 0.0
    bsdf.inputs["Subsurface Weight"].default_value = 0.15
    return mat


A = np.array(fit["base_points"])
EYE_X = float(os.environ.get("BASE_EYE_X", "0.0323"))  # half the base head's eye gap, from prepare_base.py
BASE_EYES = {"R": np.array((-EYE_X, 0.0, 0.0)), "L": np.array((EYE_X, 0.0, 0.0))}
for side, corners in (("R", (33, 133)), ("L", (362, 263))):
    # The eye corners are the most reliable eye points. Where the base head's eyeball sits relative
    # to its corners carries over, scaled by how wide the subject's eye opening is.
    corner = (X[index[corners[0]]] + X[index[corners[1]]]) / 2
    base_corner = (A[index[corners[0]]] + A[index[corners[1]]]) / 2
    ratio = np.linalg.norm(X[index[corners[0]]] - X[index[corners[1]]]) / np.linalg.norm(A[index[corners[0]]] - A[index[corners[1]]])
    centre = Vector(corner + (BASE_EYES[side] - base_corner) * ratio)
    if side in base_eyes:
        centre = Vector(base_eyes[side][0])
    # Sit the eye a little deeper, so the lids overlap it. EYE_BACK_MM tunes this.
    centre.y += float(os.environ.get("EYE_BACK_MM", "1.5")) / 1000 * RADIUS / 0.012
    radius = base_eyes[side][1] if side in base_eyes else RADIUS
    bpy.ops.mesh.primitive_uv_sphere_add(segments=64, ring_count=32, radius=radius, location=centre)
    eye = bpy.context.active_object
    eye.name = f"eye.{side}"
    eye.data.materials.append(eye_material(f"eye_{side}"))
    bpy.ops.object.shade_smooth()
    print("EYE", side, "centre mm", tuple(round(v * 1000, 1) for v in centre), "corner gap mm",
          round(float(np.linalg.norm(X[index[corners[0]]] - X[index[corners[1]]])) * 1000, 1))
bpy.ops.wm.save_as_mainfile(filepath=out)
print("EYES_OK")
