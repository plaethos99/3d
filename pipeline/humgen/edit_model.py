"""Apply the studio's changes to a saved HG human and save it (then export_model.py makes the viewer's .glb).

Usage: blender -b <human.blend> -P pipeline/humgen/edit_model.py -- <changes.json> <out.blend>
changes.json (any of):
  keys:      {livekey name: value}   face and body sliders (HG livekeys)
  height_cm: 165                     body height (HG moves the rig too)
  age:       30                      2-70 (below 20: the child body shapes; pipeline/humgen/child)
  outfit:    "outfits\\female\\Casual\\Smart_Casual.blend"   an HG outfit (as listed by the studio)
  footwear:  "footwear\\female\\Sneakers\\HG_Leather_Sneakers_Female.blend"
  colors:    {"<garment object>": {"<colour input>": "#RRGGBB"}}   garment colours (HG's clothing control node)
Keys are set first and the rig, eyes, teeth and clothes refit once (HG's update_human_from_key_change); a new
outfit is put on after, so it's fitted to the new body. Prints EDIT_MODEL_OK.
"""
import json
import os
import sys

import bpy

from HumGen3D import Human

argv = sys.argv[sys.argv.index("--") + 1:]
changes = json.load(open(os.path.abspath(argv[0]), encoding="utf-8"))
out = os.path.abspath(argv[1])
rig = next(o for o in bpy.data.objects if o.type == "ARMATURE" and o.HG.ishuman)
human = Human.from_existing(rig)
done = []

if changes.get("keys"):
    for name, value in changes["keys"].items():
        key = human.keys.get(name)
        if key is None:
            print("NO_KEY", name)
            continue
        key.set_without_update(float(value))
    human.keys.update_human_from_key_change(bpy.context)
    done.append(f"{len(changes['keys'])} sliders")
if "age" in changes:
    human.age.set(int(changes["age"]))
    done.append(f"age {changes['age']}")
if "height_cm" in changes:
    human.height.set(float(changes["height_cm"]))
    done.append(f"height {changes['height_cm']} cm")
for kind in ("outfit", "footwear"):
    if changes.get(kind):
        getattr(human.clothing, kind).set(changes[kind])
        done.append(f"{kind} {os.path.basename(changes[kind])}")


def lin(hexstr):
    hx = hexstr.lstrip("#")
    c = [int(hx[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    return [x / 12.92 if x <= 0.04045 else ((x + 0.055) / 1.055) ** 2.4 for x in c] + [1.0]


for obj_name, inputs in (changes.get("colors") or {}).items():
    obj = bpy.data.objects.get(obj_name)
    if not obj:
        print("NO_GARMENT", obj_name)
        continue
    for mat in obj.data.materials:
        ctrl = mat and mat.node_tree and mat.node_tree.nodes.get("HG_Control")
        if not ctrl:
            continue
        for name, hexstr in inputs.items():
            if name in ctrl.inputs:
                ctrl.inputs[name].default_value = lin(hexstr)
    done.append(f"colours of {obj_name}")

bpy.ops.wm.save_as_mainfile(filepath=out)
print("EDIT_MODEL_OK", out, "; ".join(done) or "nothing to change")
