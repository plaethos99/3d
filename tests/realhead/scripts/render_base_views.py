"""Render the plain base head from several known angles, so the face tracker can find its points.

Usage: blender -b base_head.blend -P render_base_views.py -- <out_dir>
Writes view_<yaw>.png and cameras.json (each view's 3x4 projection matrix).
"""
import json
import os
import sys

import bpy

sys.path.append(os.path.dirname(__file__))
import headcam  # noqa: E402

out = sys.argv[sys.argv.index("--") + 1]
os.makedirs(out, exist_ok=True)
scene = headcam.setup_render(1024, 1024, 24)
headcam.soft_studio(scene)
cam = headcam.camera(scene)


def material(name, colour, roughness):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Base Color"].default_value = (*colour, 1)
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


skin = material("skin", (0.55, 0.36, 0.27), 0.5)
sclera = material("sclera", (0.85, 0.83, 0.8), 0.2)
iris = material("iris", (0.08, 0.05, 0.03), 0.2)
def whole_eye():
    """White eyeball with a dark iris and pupil facing forward, from the eye's own bounding box,
    so the face tracker can find the eyelids around it."""
    mat = bpy.data.materials.new("whole_eye")
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    coords = nodes.new("ShaderNodeTexCoord")
    centre = nodes.new("ShaderNodeVectorMath")
    centre.operation = "SUBTRACT"
    centre.inputs[1].default_value = (0.5, 0.5, 0.5)
    links.new(coords.outputs["Generated"], centre.inputs[0])
    sep = nodes.new("ShaderNodeSeparateXYZ")
    links.new(centre.outputs[0], sep.inputs[0])
    combine = nodes.new("ShaderNodeCombineXYZ")
    links.new(sep.outputs["X"], combine.inputs["X"])
    links.new(sep.outputs["Z"], combine.inputs["Z"])
    length = nodes.new("ShaderNodeVectorMath")
    length.operation = "LENGTH"
    links.new(combine.outputs[0], length.inputs[0])
    in_front = nodes.new("ShaderNodeMath")
    in_front.operation = "LESS_THAN"
    links.new(sep.outputs["Y"], in_front.inputs[0])
    in_front.inputs[1].default_value = 0.0
    radial = nodes.new("ShaderNodeMath")
    radial.operation = "ADD"
    links.new(length.outputs["Value"], radial.inputs[0])
    flip = nodes.new("ShaderNodeMath")
    flip.operation = "SUBTRACT"
    flip.inputs[0].default_value = 1.0
    links.new(in_front.outputs[0], flip.inputs[1])
    links.new(flip.outputs[0], radial.inputs[1])
    ramp = nodes.new("ShaderNodeValToRGB")
    els = ramp.color_ramp.elements
    els[0].position, els[0].color = 0.0, (0.01, 0.01, 0.01, 1)
    els[1].position, els[1].color = 0.08, (0.12, 0.06, 0.02, 1)
    edge = els.new(0.2)
    edge.color = (0.05, 0.03, 0.02, 1)
    white = els.new(0.23)
    white.color = (0.8, 0.78, 0.75, 1)
    links.new(radial.outputs[0], ramp.inputs[0])
    links.new(ramp.outputs[0], nodes["Principled BSDF"].inputs["Base Color"])
    return mat


eye_mat = whole_eye()
for obj in bpy.data.objects:
    if obj.type != "MESH":
        continue
    obj.data.materials.clear()
    if obj.name.startswith("eye."):
        obj.data.materials.append(eye_mat)
        continue
    obj.data.materials.append(skin if obj.name == "head" else sclera if obj.name.startswith("sclera") else iris)
    if obj.name == "head":
        for poly in obj.data.polygons:
            poly.use_smooth = True

cameras = {}
for yaw in (-45, -25, 0, 25, 45):
    headcam.place_orbit(cam, yaw, 3, 0.95)
    bpy.context.view_layer.update()
    path = os.path.join(out, f"view_{yaw:+d}.png")
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)
    cameras[f"{yaw:+d}"] = {"image": path, "projection": headcam.projection(scene, cam).tolist(),
                            "location": list(cam.location)}
with open(os.path.join(out, "cameras.json"), "w", encoding="utf-8") as handle:
    json.dump(cameras, handle)
print("BASE_VIEWS_OK", len(cameras))
