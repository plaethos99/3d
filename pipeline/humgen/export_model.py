"""Export a saved HG human for the web viewer: a .glb with its face and body sliders as morph targets.

Usage: blender -b <human.blend> -P pipeline/humgen/export_model.py -- <out_dir> [--tex 2048] [--samples 4]
Writes <out_dir>/model.glb and model.json:
  sliders: [{key, label, group, part ("face"|"body"|"age"), value (current), min, max, morph}], the morph target
           names in the .glb are "m:<key>"; a slider at value v shows the morph at weight v - current
  objects: which mesh in the .glb is the body / clothes / hair (clothes can't follow body morphs live: the
           viewer hides them while body sliders are off their applied values; Apply rebuilds them)
Steps: hair (head, brows, lashes) to hair cards (a browser can't draw Blender's strand hair); skin, eyes and clothes
baked to textures (Blender's procedural shaders don't export); the body's shape keys mixed into its base shape;
each HG livekey (face, body, child ages) added back as a morph relative to that base; glTF export with skin and
morphs. Prints EXPORT_MODEL_OK.
"""
import json
import os
import re
import sys
import time

import bpy
import numpy as np

from HumGen3D import Human
from HumGen3D.backend import get_prefs

argv = sys.argv[sys.argv.index("--") + 1:]
out = os.path.abspath(argv[0])
tex = int(argv[argv.index("--tex") + 1]) if "--tex" in argv else 2048
samples = int(argv[argv.index("--samples") + 1]) if "--samples" in argv else 4
os.makedirs(out, exist_ok=True)
t0 = time.time()
ctx = bpy.context

rig = next(o for o in bpy.data.objects if o.type == "ARMATURE" and o.HG.ishuman)
human = Human.from_existing(rig)
body = human.objects.body
gender = human.gender

# --- the sliders: every livekey for this gender, with its current value ---
GROUPS = {"u_skull": "Forehead & brow", "eyes": "Eyes", "nose": "Nose", "cheeks": "Cheeks", "mouth": "Lips & mouth",
          "jaw": "Jaw", "chin": "Chin", "ears": "Ears", "l_skull": "Mouth area", "special": "Eyes",
          "Torso": "Torso", "Arms": "Arms", "Legs": "Legs", "Head": "Neck", "Muscles": "Muscles", "main": "Build",
          "Special": "Style", "age_child": "Age"}
sliders = []
for key in human.keys.all_livekeys:
    sub = key.subcategory
    if key.category == "face_proportions":
        part, group = "face", GROUPS.get(sub, sub.title())
    elif key.category == "face_presets":
        part, group = "face", "Face blend"  # the captured likeness (face_<name>) and HG's ethnic/variation faces
    elif key.category == "body_proportions":
        part, group = "body", GROUPS.get(sub, sub.title())
    elif key.category == "special" and sub == "age_child":
        part, group = "age", "Age"
    else:
        continue
    if key.name in ("height_150", "height_200"):
        continue  # height is a slider of its own (it also moves the rig)
    label = re.sub(r"[_]+", " ", key.name).strip()
    if key.name.startswith("face_") and key.category == "face_presets":
        label = "This person's face (likeness)"
    label = label[:1].upper() + label[1:]
    # ranges: HG's own panel lets face shapes go -2..2 (hard limit 10); a little more to tell faces apart
    if part == "face" and group != "Face blend":
        lo, hi = -3.0, 3.0
    elif group == "Face blend":
        lo, hi = 0.0, 2.0
    elif part == "age":
        lo, hi = 0.0, 1.0
    elif group == "Muscles":
        lo, hi = 0.0, 2.0
    elif group == "Build":
        lo, hi = 0.0, 1.5
    else:
        lo, hi = -2.0, 2.0
    sliders.append({"key": key.name, "label": label, "group": group, "part": part,
                    "value": round(float(key.value), 4), "min": min(lo, float(key.value)), "max": max(hi, float(key.value)),
                    "morph": f"m:{key.name}", "path": key.path})
print("SLIDERS", len(sliders), {p: sum(s["part"] == p for s in sliders) for p in ("face", "body", "age")})

# --- garments and their colour inputs (before baking replaces their materials), for the costume panel ---
SKIP = {"Diffuse", "Roughness", "Normal", "Mask", "Pattern", "PC1", "PC2", "PC3", "None"}


def hexof(c):
    s_ = [x * 12.92 if x <= 0.0031308 else 1.055 * x ** (1 / 2.4) - 0.055 for x in c[:3]]
    return "#%02X%02X%02X" % tuple(int(round(min(max(x, 0), 1) * 255)) for x in s_)


garments = []
for kind in ("outfit", "footwear"):
    for o in getattr(human.clothing, kind).objects:
        colours = []
        for mat in o.data.materials:
            ctrl = mat and mat.node_tree and mat.node_tree.nodes.get("HG_Control")
            if ctrl:
                for i in ctrl.inputs:
                    if i.type == "RGBA" and i.name not in SKIP and not i.is_linked:
                        colours.append({"input": i.name, "label": i.name.replace("_C0", "").replace("_", " "),
                                        "hex": hexof(list(i.default_value))})
        garments.append({"object": o.name, "kind": kind, "label": o.name.replace("HG_", "").split(".")[0].replace("_", " "),
                         "colors": colours})
outfit_now = {"outfit": getattr(human.clothing.outfit, "_active", None),
              "footwear": getattr(human.clothing.footwear, "_active", None)}

# --- hair to hair cards ---
hair_objs = []
for name in ("regular_hair", "eyebrows", "eyelashes"):
    setting = getattr(human.hair, name, None)
    if setting is None or not setting.modifiers:
        continue
    try:
        before = set(bpy.data.objects)
        setting.convert_to_haircards(quality="high", context=ctx)
        new = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
        hair_objs += new
        print("HAIRCARDS", name, [o.name for o in new])
    except Exception as e:  # noqa: BLE001
        print("HAIRCARDS_FAILED", name, repr(e)[:300])
for m in body.modifiers:  # strand hair left on the body would not export anyway
    if m.type == "PARTICLE_SYSTEM":
        m.show_viewport = m.show_render = False
print(f"hair done ({time.time() - t0:.0f}s)")

# --- textures ---
bk = human.process.baking
bk.resolution_body = tex
bk.resolution_clothes = tex  # hair cards bake at this size too
bk.resolution_eyes = max(256, tex // 4)
bake_dir = os.path.join(out, "textures")
os.makedirs(bake_dir, exist_ok=True)
bk.bake_all(bake_dir, samples, context=ctx)
print(f"bake done ({time.time() - t0:.0f}s)")

# --- body: shape keys mixed into the base, then the sliders as morphs relative to it ---
me = body.data
kb = me.shape_keys.key_blocks
N = len(me.vertices)
mix = np.empty(N * 3)
kb["Basis"].data.foreach_get("co", mix)
basis = mix.copy()
for k in kb[1:]:
    if k.mute or not k.value:
        continue
    c = np.empty(N * 3)
    k.data.foreach_get("co", c)
    rel = kb[k.relative_key.name] if k.relative_key else kb["Basis"]
    r = np.empty(N * 3)
    rel.data.foreach_get("co", r)
    mix += k.value * (c - r)
body.shape_key_clear()
me.vertices.foreach_set("co", mix)
body.shape_key_add(name="Basis", from_mix=False)
content = get_prefs().filepath
for s in sliders:
    d = np.load(os.path.join(content, s["path"]))
    off = np.zeros(N * 3)
    off[d["indices"]] = d["relative_coordinates"]
    sk = body.shape_key_add(name=s["morph"], from_mix=False)
    sk.data.foreach_set("co", mix + off)
    sk.value = 0.0
    del s["path"]
me.update()
print(f"morphs done ({time.time() - t0:.0f}s)")

# the skin HG hides under the clothes (its mask modifiers, which don't export) as a vertex colour, red = hidden:
# the viewer skips drawing it while the clothes are on (and draws it when it hides the clothes)
hidden = np.zeros(N, bool)
for mod in body.modifiers:
    if mod.type == "MASK" and mod.vertex_group in body.vertex_groups:
        gi = body.vertex_groups[mod.vertex_group].index
        ing = np.array([any(g.group == gi and g.weight > 0 for g in v.groups) for v in me.vertices])
        hidden |= ing if mod.invert_vertex_group else ~ing
attr = me.color_attributes.new("hide", "FLOAT_COLOR", "POINT")
cols = np.zeros((N, 4), np.float32)
cols[:, 3] = 1.0
cols[hidden, 0] = 1.0
attr.data.foreach_set("color", cols.ravel())
me.color_attributes.active_color = attr
print("HIDDEN under clothes:", int(hidden.sum()), "of", N, "points")

# eyes: the see-through outer shell doesn't export well
eyes = human.objects.eyes
if eyes.data.materials and eyes.data.materials[0] and "Outer" in eyes.data.materials[0].name:
    eyes.data.materials.pop(index=0)

# --- export ---
clothes = list(human.clothing.outfit.objects) + list(human.clothing.footwear.objects)
for o in clothes:  # their own shape keys (body proportions, correctives) mixed in: what's shown now
    if o.data.shape_keys:
        ctx.view_layer.objects.active = o
        for x in ctx.selected_objects:
            x.select_set(False)
        o.select_set(True)
        bpy.ops.object.shape_key_remove(all=True, apply_mix=True)
col = bpy.data.collections.new("Export")
ctx.scene.collection.children.link(col)
parts = [rig, body, eyes, human.objects.upper_teeth, human.objects.lower_teeth] + clothes + hair_objs
for o in parts:
    if o.name not in col.objects:
        col.objects.link(o)
ctx.view_layer.active_layer_collection = ctx.view_layer.layer_collection.children[col.name]
glb = os.path.join(out, "model.glb")
# vertex colours (the hidden-skin marker): the option's name changed across Blender versions
props = bpy.ops.export_scene.gltf.get_rna_type().properties
colour_arg = {}
if "export_vertex_color" in props:
    colour_arg["export_vertex_color"] = "ACTIVE"
elif "export_colors" in props:
    colour_arg["export_colors"] = True
bpy.ops.export_scene.gltf(filepath=glb, export_format="GLB", use_active_collection=True, export_morph=True,
                          export_skins=True, export_animations=False, export_image_format="AUTO",
                          export_morph_normal=False, **colour_arg)
age_now = human.objects.body.get("Age", 30) if hasattr(human.objects.body, "get") else 30
info = {"gender": gender, "sliders": sliders, "height_cm": round(human.height.centimeters, 1),
        "age": int(age_now), "garments": garments, "current": outfit_now,
        "objects": {"body": body.name, "clothes": [o.name for o in clothes], "hair": [o.name for o in hair_objs],
                    "eyes": eyes.name}}
json.dump(info, open(os.path.join(out, "model.json"), "w"), indent=1)
print(f"EXPORT_MODEL_OK {glb} {os.path.getsize(glb) / 1e6:.1f} MB ({time.time() - t0:.0f}s)")
