"""Repaint finished eyes with a layered iris: pupil, inner iris, outer band, dark rim, whites.

Usage: blender -b head.blend -P tune_eyes.py -- <out.blend> <iris_share> <pupil_share> <inner_rgb> <outer_rgb> <white_rgb> [specular]

iris_share is the iris radius as a share of the eyeball radius, pupil_share the pupil's as a share
of the iris. Colours are linear RGB. Many irises, drawn or real, are darker near the pupil and
lighter toward a band near the edge, then end in a thin dark ring (the limbal ring); that ring and
a crisp pupil are much of what makes eyes read as clear rather than cloudy.
"""
import sys

import bpy

args = sys.argv[sys.argv.index("--") + 1:]
out = args[0]
iris_share, pupil_share = float(args[1]), float(args[2])
inner, outer, white = (tuple(float(v) for v in a.split(",")) for a in args[3:6])
specular = float(args[6]) if len(args) > 6 else 0.12
for obj in bpy.data.objects:
    if not obj.name.startswith("eye.") or not obj.data.materials:
        continue
    radius = obj.dimensions.x / 2
    span = radius / 0.6  # the ramp in add_eyes.py runs over 0..span of distance from the eye's axis
    iris = radius * iris_share / span
    pupil = iris * pupil_share
    mat = obj.data.materials[0]
    ramp = next(n for n in mat.node_tree.nodes if n.type == "VALTORGB").color_ramp
    while len(ramp.elements) > 1:
        ramp.elements.remove(ramp.elements[-1])
    ramp.elements[0].position, ramp.elements[0].color = 0.0, (0.004, 0.003, 0.003, 1)
    stops = [
        (pupil, (0.004, 0.003, 0.003)),
        (pupil + 0.006, inner),
        (pupil + (iris - pupil) * 0.45, inner),
        (iris * 0.86, outer),
        (iris * 0.93, outer),
        (iris * 0.965, (0.012, 0.006, 0.003)),
        (iris + 0.004, (0.012, 0.006, 0.003)),
        (iris + 0.03, white),
        (0.6, (white[0] * 0.8, white[1] * 0.72, white[2] * 0.7)),
    ]
    for position, colour in stops:
        element = ramp.elements.new(min(position, 0.999))
        element.color = (*colour, 1)
    bsdf = mat.node_tree.nodes["Principled BSDF"]
    bsdf.inputs["Specular IOR Level"].default_value = specular
    bsdf.inputs["Roughness"].default_value = 0.1
    print("EYE", obj.name, "radius mm", round(radius * 1000, 1), "iris mm", round(radius * iris_share * 1000, 1))
bpy.ops.wm.save_as_mainfile(filepath=out)
print("EYES_TUNED")
