"""Rebuild a named expression shape from a mix of the face rig's shapes.

Usage: blender -b character.blend -P set_expression_mix.py -- <out.blend> <name> shape=amount [...]
e.g. surprised jaw_open=1.1 mouth_O=0.3 brows_up=1 smile=-0.5 eyes_wide=1
The mouth parts (tongue, teeth) get the same share of jaw_open.
"""
import json
import sys

import bpy
import numpy as np

args = sys.argv[sys.argv.index("--") + 1:]
out, name = args[0], args[1]
mix = {k: float(v) for k, v in (a.split("=") for a in args[2:])}
for obj in bpy.data.objects:
    keys = getattr(obj.data, "shape_keys", None) if obj.data is not None else None
    if keys is None or "jaw_open" not in keys.key_blocks:
        continue
    base = np.array([v.co[:] for v in obj.data.vertices])
    total = base.copy()
    for shape, amount in mix.items():
        if shape in keys.key_blocks and shape != name:
            total += amount * (np.array([d.co[:] for d in keys.key_blocks[shape].data]) - base)
    block = keys.key_blocks.get(name) or obj.shape_key_add(name=name)
    block.data.foreach_set("co", total.ravel())
    if obj.name == "head":
        obj["expression_mix_" + name] = json.dumps(mix)
# New shapes must not be left switched on: save the character with a neutral face.
for _obj in bpy.data.objects:
    _keys = getattr(_obj.data, "shape_keys", None) if _obj.data is not None else None
    if _keys is not None:
        for _block in _keys.key_blocks[1:]:
            _block.value = 0.0
bpy.ops.wm.save_as_mainfile(filepath=out)
print("MIX_SET", name, mix)
