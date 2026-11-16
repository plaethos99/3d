"""Bring Joyce's clothes closer to her reference: real wool, ribbing, tapered trousers and folds.

Usage: blender -b joyce.blend -P clothes_detail.py -- <out.blend> [KEY=value ...]

Knit: stitches 1.6x smaller (as in the reference), colours softened and heathered like wool spun
from mixed fibres, a fuzzy sheen and fibre bumps. Ribbing: stronger rib relief on cuffs, hem, front
band and pocket trims. Blouse and trousers: woven-fabric texture and a soft sheen. Folds: soft
creases at the inside of the elbows and at the knees and ankles, as relief in the shading. Trousers
taper toward the ankle.
"""
import sys

import bpy
import numpy as np

args = sys.argv[sys.argv.index("--") + 1:]
out = args[0]
opt = dict(a.split("=", 1) for a in args[1:])
STITCH_SCALE = float(opt.get("STITCH_SCALE", 1.6))
SATURATION = float(opt.get("SATURATION", 0.72))
TAPER = float(opt.get("TAPER", 0.80))
SHEEN = float(opt.get("SHEEN", 0.8))
FOLDS = opt.get("FOLDS", "1") == "1"


def nodes_of(name):
    mat = bpy.data.materials[name]
    return mat, mat.node_tree.nodes, mat.node_tree.links


def bump_chain(nodes, links, bsdf, height_socket, strength, distance):
    """Add a bump on top of whatever normal the BSDF already has."""
    bump = nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = strength
    bump.inputs["Distance"].default_value = distance
    links.new(height_socket, bump.inputs["Height"])
    existing = bsdf.inputs["Normal"].links
    if existing:
        links.new(existing[0].from_socket, bump.inputs["Normal"])
    links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    return bump


def wool(mat_name, scale_images, heather_scale):
    mat, nodes, links = nodes_of(mat_name)
    bsdf = nodes["Principled BSDF"]
    coords = nodes.new("ShaderNodeTexCoord")
    if scale_images != 1.0:
        mapping = nodes.new("ShaderNodeMapping")
        mapping.inputs["Scale"].default_value = (scale_images, scale_images, 1)
        links.new(coords.outputs["UV"], mapping.inputs["Vector"])
        for node in nodes:
            if node.type == "TEX_IMAGE":
                links.new(mapping.outputs["Vector"], node.inputs["Vector"])
    colour_link = bsdf.inputs["Base Color"].links
    if colour_link:
        source = colour_link[0].from_socket
        soften = nodes.new("ShaderNodeHueSaturation")
        soften.inputs["Saturation"].default_value = SATURATION
        soften.inputs["Value"].default_value = 0.95
        links.new(source, soften.inputs["Color"])
        heather = nodes.new("ShaderNodeTexNoise")
        heather.inputs["Scale"].default_value = heather_scale
        heather.inputs["Detail"].default_value = 6
        links.new(coords.outputs["Object"], heather.inputs["Vector"])
        tone = nodes.new("ShaderNodeMapRange")
        tone.inputs["To Min"].default_value = 0.84
        tone.inputs["To Max"].default_value = 1.08
        links.new(heather.outputs["Fac"], tone.inputs["Value"])
        mixed = nodes.new("ShaderNodeMix")
        mixed.data_type = "RGBA"
        mixed.blend_type = "MULTIPLY"
        mixed.inputs["Factor"].default_value = 1.0
        links.new(soften.outputs["Color"], mixed.inputs["A"])
        links.new(tone.outputs["Result"], mixed.inputs["B"])
        links.new(mixed.outputs["Result"], bsdf.inputs["Base Color"])
    bsdf.inputs["Roughness"].default_value = 0.95
    bsdf.inputs["Sheen Weight"].default_value = SHEEN
    bsdf.inputs["Sheen Roughness"].default_value = 0.45
    bsdf.inputs["Specular IOR Level"].default_value = 0.15
    fibres = nodes.new("ShaderNodeTexNoise")
    fibres.inputs["Scale"].default_value = 900.0
    fibres.inputs["Detail"].default_value = 3
    links.new(coords.outputs["Object"], fibres.inputs["Vector"])
    bump_chain(nodes, links, bsdf, fibres.outputs["Fac"], 0.25, 0.0008)
    for node in nodes:
        if node.type == "BUMP" and node is not nodes.get("Bump.001"):
            pass
    return mat


wool("knit", STITCH_SCALE, 160.0)
wool("band", 1.0, 160.0)
band = bpy.data.materials["band"].node_tree.nodes
if "Bump" in band:
    band["Bump"].inputs["Strength"].default_value = 1.0
    band["Bump"].inputs["Distance"].default_value = 0.006
knit = bpy.data.materials["knit"].node_tree.nodes
if "Bump" in knit:
    knit["Bump"].inputs["Strength"].default_value = 0.9


def woven(mat_name, sheen, weave_scale, roughness):
    mat, nodes, links = nodes_of(mat_name)
    bsdf = nodes["Principled BSDF"]
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Sheen Weight"].default_value = sheen
    bsdf.inputs["Sheen Roughness"].default_value = 0.4
    bsdf.inputs["Specular IOR Level"].default_value = 0.2
    coords = nodes.new("ShaderNodeTexCoord")
    weave = nodes.new("ShaderNodeTexWave")
    weave.wave_type = "BANDS"
    weave.inputs["Scale"].default_value = weave_scale
    weave.inputs["Distortion"].default_value = 1.5
    links.new(coords.outputs["Object"], weave.inputs["Vector"])
    bump_chain(nodes, links, bsdf, weave.outputs["Fac"], 0.08, 0.0004)
    return mat, nodes, links, bsdf, coords


def folds(mat_name, mask_ranges, axis_scale, strength):
    """Soft creases across a garment, where it bunches: mask_ranges are (low, high) heights."""
    mat = bpy.data.materials[mat_name]
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes["Principled BSDF"]
    coords = next((n for n in nodes if n.type == "TEX_COORD"), None) or nodes.new("ShaderNodeTexCoord")
    wave = nodes.new("ShaderNodeTexWave")
    wave.wave_type = "BANDS"
    wave.bands_direction = "Z"
    wave.inputs["Scale"].default_value = axis_scale
    wave.inputs["Distortion"].default_value = 6.0
    wave.inputs["Detail"].default_value = 2.0
    wave.inputs["Detail Scale"].default_value = 1.5
    links.new(coords.outputs["Object"], wave.inputs["Vector"])
    sep = nodes.new("ShaderNodeSeparateXYZ")
    links.new(coords.outputs["Object"], sep.inputs[0])
    total = None
    for low, high in mask_ranges:
        rise = nodes.new("ShaderNodeMapRange")
        rise.inputs["From Min"].default_value = low
        rise.inputs["From Max"].default_value = (low + high) / 2
        fall = nodes.new("ShaderNodeMapRange")
        fall.inputs["From Min"].default_value = high
        fall.inputs["From Max"].default_value = (low + high) / 2
        links.new(sep.outputs["Z"], rise.inputs["Value"])
        links.new(sep.outputs["Z"], fall.inputs["Value"])
        both = nodes.new("ShaderNodeMath")
        both.operation = "MINIMUM"
        links.new(rise.outputs["Result"], both.inputs[0])
        links.new(fall.outputs["Result"], both.inputs[1])
        if total is None:
            total = both
        else:
            join = nodes.new("ShaderNodeMath")
            join.operation = "MAXIMUM"
            links.new(total.outputs[0], join.inputs[0])
            links.new(both.outputs[0], join.inputs[1])
            total = join
    masked = nodes.new("ShaderNodeMath")
    masked.operation = "MULTIPLY"
    links.new(wave.outputs["Fac"], masked.inputs[0])
    links.new(total.outputs[0], masked.inputs[1])
    bump_chain(nodes, links, bsdf, masked.outputs[0], strength, 0.01)


woven("blouse", 0.4, 900.0, 0.65)
woven("pants", 0.35, 1400.0, 0.72)
# Knees and ankles for the trousers, elbows for the sleeves (heights in metres, rest pose).
if FOLDS:
    folds("pants", [(0.30, 0.46), (0.05, 0.20)], 18.0, 0.5)
    folds("knit", [(0.70, 0.82)], 14.0, 0.35)

# Taper the trouser legs toward the ankle, around each leg's own axis.
for side in "LR":
    leg = bpy.data.objects[f"leg_{side}"]
    co = np.array([v.co for v in leg.data.vertices])
    z = co[:, 2]
    top, bottom = 0.60, 0.08
    factor = 1.0 - (1.0 - TAPER) * np.clip((top - z) / (top - bottom), 0, 1)
    for k in np.unique(np.round(z, 3)):
        pass
    # The leg's axis at each height: the middle of the vertices at that height.
    order = np.argsort(z)
    centre = np.zeros_like(co)
    window = 40
    for i, idx in enumerate(order):
        lo, hi = max(i - window, 0), min(i + window, len(order))
        centre[idx] = co[order[lo:hi]].mean(axis=0)
    co[:, :2] = centre[:, :2] + (co[:, :2] - centre[:, :2]) * factor[:, None]
    leg.data.vertices.foreach_set("co", co.ravel())
    leg.data.update()
print("TAPERED trousers to", TAPER, "at the ankle")
bpy.ops.wm.save_as_mainfile(filepath=out)
print("CLOTHES_DETAIL_OK")
