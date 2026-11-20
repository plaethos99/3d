"""Export a saved HG human for the web viewer: a .glb with its face and body sliders as morph targets.

Usage: blender -b <human.blend> -P pipeline/humgen/export_model.py -- <out_dir> [--tex 2048] [--samples 4]
Writes <out_dir>/model.glb and model.json:
  sliders: [{key, label, group, part ("face"|"body"|"age"), value (current), min, max, morph}], the morph target
           names in the .glb are "m:<key>"; a slider at value v shows the morph at weight v - current
  objects: which mesh in the .glb is the body / clothes / hair (clothes can't follow body morphs live: the
           viewer hides them while body sliders are off their applied values; Apply rebuilds them)
Steps: skin, eyes and clothes baked to textures (Blender's procedural shaders don't export); hair (head, brows) to
hair cards with HG's hair-card texture sheets tinted to the hair colour (a browser can't draw strand hair); the body's shape keys mixed into its base shape;
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
          "Special": "Style", "age_child": "Age", "flame": "Face shape (FLAME)"}
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
    if key.name.startswith("flame_shape_"):  # 1.0 = 2 standard deviations of real faces: +-1.5 is already rare
        label = "Shape " + key.name.rsplit("_", 1)[1]
        lo, hi = -1.5, 1.5
    elif part == "face" and group != "Face blend":
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

# --- textures ---
bk = human.process.baking
bk.resolution_body = tex
bk.resolution_clothes = tex
bk.resolution_eyes = max(256, tex // 4)
bake_dir = os.path.join(out, "textures")
os.makedirs(bake_dir, exist_ok=True)
bk.bake_all(bake_dir, samples, context=ctx)
print(f"bake done ({time.time() - t0:.0f}s)")

# --- hair to hair cards (after the bake: HG's hair-card shader doesn't bake cleanly) ---
# Each card gets a plain material a browser can draw: HG's own hair-card texture sheets (strand transparency,
# shading, normals: content/hair/haircards), tinted to the character's hair colour, with a hard alpha cut-off.
HC_DIR = os.path.join(get_prefs().filepath, "hair", "haircards")


def hair_linear_rgb(setting):
    """Roughly the colour HG's hair shader gives for its lightness and redness."""
    L = min(max(float(setting.lightness.value), 0.0), 1.0)
    R = min(max(float(setting.redness.value), 0.0), 1.0)
    dark, blond, red = np.array([0.012, 0.009, 0.007]), np.array([0.62, 0.45, 0.24]), np.array([0.30, 0.09, 0.03])
    c = dark + (blond - dark) * L ** 1.4
    return c * (1 - 0.45 * R) + red * (0.45 * R) * (0.25 + L)


def load_image(name):
    path = os.path.join(HC_DIR, name)
    return bpy.data.images.load(path, check_existing=True) if os.path.exists(path) else None


def hair_card_material(rgb):
    ao, alpha, normal = (load_image(f"HairMediumLength_{k}.png") for k in ("AO", "ALPHA", "NORMAL"))
    w, h = alpha.size
    A = np.empty(w * h * 4, np.float32)
    alpha.pixels.foreach_get(A)
    if tuple(ao.size) != (w, h):
        ao.scale(w, h)
    O = np.empty(w * h * 4, np.float32)
    ao.pixels.foreach_get(O)
    shade = O.reshape(-1, 4)[:, :1] * 0.8 + 0.2
    lin = np.clip(shade * rgb[None, :], 0, 1)
    srgb = np.where(lin <= 0.0031308, lin * 12.92, 1.055 * np.power(lin, 1 / 2.4) - 0.055)
    img = bpy.data.images.new("hair_cards_colour", w, h, alpha=True, float_buffer=False)
    img.colorspace_settings.name = "sRGB"
    img.pixels.foreach_set(np.concatenate([srgb, A.reshape(-1, 4)[:, :1]], 1).astype(np.float32).ravel())
    os.makedirs(out_dir_tex, exist_ok=True)
    img.filepath_raw = os.path.join(out_dir_tex, "hair_cards_colour.png")
    img.file_format = "PNG"
    img.save()
    mat = bpy.data.materials.new("Hair_Cards")
    nt = mat.node_tree
    bsdf = nt.nodes["Principled BSDF"]
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = img
    nt.links.new(t.outputs["Color"], bsdf.inputs["Base Color"])
    cut = nt.nodes.new("ShaderNodeMath")  # a "greater than" on alpha: glTF's MASK mode (hard cut-off)
    cut.operation = "GREATER_THAN"
    cut.inputs[1].default_value = 0.35
    nt.links.new(t.outputs["Alpha"], cut.inputs[0])
    nt.links.new(cut.outputs[0], bsdf.inputs["Alpha"])
    if normal:
        normal.colorspace_settings.name = "Non-Color"
        tn = nt.nodes.new("ShaderNodeTexImage")
        tn.image = normal
        nm = nt.nodes.new("ShaderNodeNormalMap")
        nt.links.new(tn.outputs["Color"], nm.inputs["Color"])
        nt.links.new(nm.outputs["Normal"], bsdf.inputs["Normal"])
    bsdf.inputs["Roughness"].default_value = 0.5
    mat.use_backface_culling = False
    return mat


def solid_material(name, rgb):
    mat = bpy.data.materials.new(name)
    b = mat.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Roughness"].default_value = 0.6
    return mat


out_dir_tex = os.path.join(out, "textures")
hair_objs = []
hair_rgb = hair_linear_rgb(human.hair.regular_hair)
# hair length / waves set in the studio: long_hair.py grows the style's strands into cards; HG's own conversion
# below then keeps only its scalp cap
hair_len_cm = float(rig.get("studio_hair_length", 0) or 0)
hair_wave = float(rig.get("studio_hair_wave", 0) or 0)
long_mode = hair_len_cm > 0 or hair_wave > 0
if long_mode:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import long_hair
    from indian import body_world

    Pb, Nb = body_world(human)
    hair_objs.append(long_hair.build(human, hair_len_cm / 100, hair_wave, hair_rgb, out_dir_tex, Pb, Nb))
for name in ("regular_hair", "eyebrows"):
    setting = getattr(human.hair, name, None)
    if setting is None or not any(m.show_viewport for m in setting.modifiers):
        continue
    try:
        before = set(bpy.data.objects)
        setting.convert_to_haircards(quality="high", context=ctx)
        new = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
        for o in new:
            for i, slot in enumerate(o.material_slots):
                mname = slot.material.name if slot.material else ""
                if "Haircards" in mname:
                    o.material_slots[i].material = hair_card_material(hair_rgb)
                else:  # the cap under the hair: the scalp coloured like the hair
                    o.material_slots[i].material = solid_material("Hair_Cap", hair_rgb * 0.8)
        for o in new:  # HG's cards carry a colour layer with no alpha: in a viewer it would cut the strands away
            for a in list(o.data.color_attributes):
                o.data.color_attributes.remove(a)
            if long_mode and name == "regular_hair":  # keep only HG's scalp cap: the long cards replace its cards
                import bmesh as _bm
                card_slots = {i for i, sl in enumerate(o.material_slots) if sl.material and sl.material.name.startswith("Hair_Cards")}
                b = _bm.new()
                b.from_mesh(o.data)
                _bm.ops.delete(b, geom=[f for f in b.faces if f.material_index in card_slots], context="FACES")
                b.to_mesh(o.data)
                b.free()
        hair_objs += new
        print("HAIRCARDS", name, [o.name for o in new])
    except Exception as e:  # noqa: BLE001
        print("HAIRCARDS_FAILED", name, repr(e)[:300])
for m in body.modifiers:  # strand hair left on the body would not export anyway
    if m.type == "PARTICLE_SYSTEM":
        m.show_viewport = m.show_render = False
print(f"hair done ({time.time() - t0:.0f}s)")

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
# hair cards (brows, the scalp cap, hair roots) follow the face sliders: each card point takes the move of the
# nearest skin point, fully within 1.5 cm of the skin, fading to none at 8 cm (long strands don't stretch).
# Without it a reshaped face sinks the brows into the skin or leaves them floating.
from mathutils.kdtree import KDTree  # noqa: E402

Mb = np.array(body.matrix_world)
Pw = (Mb[:3, :3] @ mix.reshape(-1, 3).T).T + Mb[:3, 3]
tree = KDTree(N)
for i, p in enumerate(Pw):
    tree.insert(p, i)
tree.balance()
face_sliders = [s for s in sliders if s["part"] == "face"]
for o in hair_objs:
    if o.type != "MESH" or not len(o.data.vertices):
        continue
    if o.data.shape_keys:
        o.shape_key_clear()
    n = len(o.data.vertices)
    co = np.empty(n * 3)
    o.data.vertices.foreach_get("co", co)
    Mo = np.array(o.matrix_world)
    W = (Mo[:3, :3] @ co.reshape(-1, 3).T).T + Mo[:3, 3]
    near = np.empty(n, int)
    dist = np.empty(n)
    for i, p in enumerate(W):
        _, near[i], dist[i] = tree.find(p)
    w = np.clip((0.08 - dist) / (0.08 - 0.015), 0, 1)
    if not w.any():
        continue
    to_local = np.linalg.inv(Mo[:3, :3]) @ Mb[:3, :3]
    o.shape_key_add(name="Basis", from_mix=False)
    for s in face_sliders:
        off = body.data.shape_keys.key_blocks[s["morph"]]
        c = np.empty(N * 3)
        off.data.foreach_get("co", c)
        D = (c - mix).reshape(-1, 3)[near] * w[:, None]
        if not np.abs(D).max() > 1e-6:
            continue
        sk = o.shape_key_add(name=s["morph"], from_mix=False)
        sk.data.foreach_set("co", co + (to_local @ D.T).T.ravel())
        sk.value = 0.0
    print("HAIR MORPHS", o.name, len(o.data.shape_keys.key_blocks) - 1, "of", len(face_sliders))
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
        "hair": {"style": getattr(human.hair.regular_hair, "_active", None),
                 "length_cm": round(float(rig.get("studio_hair_length", 0) or 0), 1),
                 "wave": round(float(rig.get("studio_hair_wave", 0) or 0), 2),
                 "lightness": round(float(human.hair.regular_hair.lightness.value), 3),
                 "redness": round(float(human.hair.regular_hair.redness.value), 3)},
        "objects": {"body": body.name, "clothes": [o.name for o in clothes], "hair": [o.name for o in hair_objs],
                    "eyes": eyes.name}}
json.dump(info, open(os.path.join(out, "model.json"), "w"), indent=1)
print(f"EXPORT_MODEL_OK {glb} {os.path.getsize(glb) / 1e6:.1f} MB ({time.time() - t0:.0f}s)")
