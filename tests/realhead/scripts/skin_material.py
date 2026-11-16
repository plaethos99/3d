"""Give the head a realistic skin material from the projected textures.

Usage: blender -b head_eyes.blend -P skin_material.py -- <textures_dir> <out.blend>
Skin lets light travel a few millimetres under its surface before it comes back out, mostly red
light, which is what keeps real skin from looking like painted plastic. Hair is rougher and has
no such glow. Fine bumps come from the photo's own small-scale detail.
"""
import os
import sys

import bpy

tex_dir, out = sys.argv[sys.argv.index("--") + 1:][:2]
head = bpy.data.objects["head"]
mat = bpy.data.materials.new("skin")
mat.use_nodes = True
nodes, links = mat.node_tree.nodes, mat.node_tree.links
bsdf = nodes["Principled BSDF"]


def image_node(name, colour_space):
    node = nodes.new("ShaderNodeTexImage")
    node.image = bpy.data.images.load(os.path.join(tex_dir, name))
    node.image.colorspace_settings.name = colour_space
    node.interpolation = "Cubic"
    return node


colour = image_node("skin_colour.png", "sRGB")
detail = image_node("skin_detail.png", "Non-Color")
hair = image_node("hair_mask.png", "Non-Color")
links.new(colour.outputs["Color"], bsdf.inputs["Base Color"])

# Roughness: skin 0.42, hair 0.9.
rough = nodes.new("ShaderNodeMapRange")
rough.inputs["To Min"].default_value = 0.42
rough.inputs["To Max"].default_value = 0.9
links.new(hair.outputs["Color"], rough.inputs["Value"])
links.new(rough.outputs["Result"], bsdf.inputs["Roughness"])

# Subsurface on skin only.
sss = nodes.new("ShaderNodeMapRange")
sss.inputs["To Min"].default_value = 0.35
sss.inputs["To Max"].default_value = 0.0
links.new(hair.outputs["Color"], sss.inputs["Value"])
links.new(sss.outputs["Result"], bsdf.inputs["Subsurface Weight"])
bsdf.inputs["Subsurface Radius"].default_value = (1.0, 0.35, 0.18)
bsdf.inputs["Subsurface Scale"].default_value = 0.004
links.new(colour.outputs["Color"], bsdf.inputs["Base Color"])
# Shine: skin has a soft sheen; the hair's tight curls scatter it away, so it barely shines.
shine = nodes.new("ShaderNodeMapRange")
shine.inputs["To Min"].default_value = 0.15
shine.inputs["To Max"].default_value = 0.02
links.new(hair.outputs["Color"], shine.inputs["Value"])
links.new(shine.outputs["Result"], bsdf.inputs["Specular IOR Level"])

bump = nodes.new("ShaderNodeBump")
bump.inputs["Strength"].default_value = 0.35
bump.inputs["Distance"].default_value = 0.0006
links.new(detail.outputs["Color"], bump.inputs["Height"])

# Short, tightly curled hair: small rounded tufts about 3 mm across, only where the hair is.
# The video is too soft to show single curls, so their size is a typical value, not measured.
coords = nodes.new("ShaderNodeTexCoord")
tufts = nodes.new("ShaderNodeTexVoronoi")
tufts.feature = "SMOOTH_F1"
tufts.inputs["Scale"].default_value = 300.0
tufts.inputs["Randomness"].default_value = 1.0
links.new(coords.outputs["Object"], tufts.inputs["Vector"])
grain = nodes.new("ShaderNodeTexNoise")
grain.inputs["Scale"].default_value = 2200.0
grain.inputs["Detail"].default_value = 4.0
links.new(coords.outputs["Object"], grain.inputs["Vector"])
mix = nodes.new("ShaderNodeMath")
mix.operation = "MULTIPLY_ADD"
mix.inputs[1].default_value = 0.35
links.new(grain.outputs["Fac"], mix.inputs[0])
links.new(tufts.outputs["Distance"], mix.inputs[2])
in_hair = nodes.new("ShaderNodeMath")
in_hair.operation = "MULTIPLY"
links.new(mix.outputs[0], in_hair.inputs[0])
links.new(hair.outputs["Color"], in_hair.inputs[1])
hair_bump = nodes.new("ShaderNodeBump")
hair_bump.inputs["Strength"].default_value = 0.3
hair_bump.inputs["Distance"].default_value = 0.0012
links.new(in_hair.outputs[0], hair_bump.inputs["Height"])
links.new(bump.outputs["Normal"], hair_bump.inputs["Normal"])
links.new(hair_bump.outputs["Normal"], bsdf.inputs["Normal"])

# Darken the hollows between tufts a little, as real hair shades itself.
shade = nodes.new("ShaderNodeMapRange")
shade.inputs["To Min"].default_value = 0.8
shade.inputs["To Max"].default_value = 1.0
links.new(tufts.outputs["Distance"], shade.inputs["Value"])
amount = nodes.new("ShaderNodeMix")
amount.data_type = "FLOAT"
links.new(hair.outputs["Color"], amount.inputs["Factor"])
amount.inputs["A"].default_value = 1.0
links.new(shade.outputs["Result"], amount.inputs["B"])
tinted = nodes.new("ShaderNodeVectorMath")
tinted.operation = "SCALE"
links.new(colour.outputs["Color"], tinted.inputs[0])
links.new(amount.outputs["Result"], tinted.inputs["Scale"])
links.new(tinted.outputs[0], bsdf.inputs["Base Color"])

head.data.materials.clear()
head.data.materials.append(mat)
for poly in head.data.polygons:
    poly.use_smooth = True
bpy.ops.wm.save_as_mainfile(filepath=out)
print("MATERIAL_OK")
