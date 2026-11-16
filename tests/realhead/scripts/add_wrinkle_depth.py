"""Turn the wrinkles and creases seen in the texture into real surface depth.

Usage: blender -b head.blend -P add_wrinkle_depth.py -- <skin_detail.png> <out.blend>

The head is subdivided so it has enough vertices to hold fine creases, then each vertex is moved
in or out along its normal by the texture's small-scale light and dark (darker = deeper fold).
A crease made this way catches light and casts a tiny shadow from any direction, which a painted
line or a bump map only fakes. DEPTH_MM sets the deepest fold.
"""
import os
import sys

import bpy

detail_path, out = sys.argv[sys.argv.index("--") + 1:][:2]
DEPTH = float(os.environ.get("DEPTH_MM", "1.2")) / 1000
LEVELS = int(os.environ.get("SUBDIV_LEVELS", "2"))

head = bpy.data.objects["head"]
subdivide = head.modifiers.new("detail_subdivision", "SUBSURF")
subdivide.levels = LEVELS
subdivide.render_levels = LEVELS
texture = bpy.data.textures.new("wrinkles", "IMAGE")
texture.image = bpy.data.images.load(detail_path)
texture.image.colorspace_settings.name = "Non-Color"
texture.extension = "EXTEND"
displace = head.modifiers.new("wrinkle_depth", "DISPLACE")
displace.texture = texture
displace.texture_coords = "UV"
displace.mid_level = 0.5
displace.strength = DEPTH * 2  # the detail map spans about +-0.5 around its middle
# A particle hair system, if any, must stay last in the stack.
hair = next((m for m in head.modifiers if m.type == "PARTICLE_SYSTEM"), None)
if hair is not None:
    bpy.context.view_layer.objects.active = head
    while head.modifiers[-1] != hair:
        bpy.ops.object.modifier_move_down(modifier=hair.name)
bpy.ops.wm.save_as_mainfile(filepath=out)
print("WRINKLE_DEPTH_OK", "levels", LEVELS, "deepest fold mm", DEPTH * 1000)
