"""Make one Human Generator human by script and render check pictures.

Usage: blender -b -P pipeline/humgen/make_human.py -- <out_dir> [--gender female|male] [--preset N|name]
       [--outfit N|name] [--footwear N|name] [--hair N|name] [--pose N|name] [--age 35] [--list]
       [--reset-body] [--child-look] [--key name=value ...] [--skin N|name] [--skin-tone x] [--skin-redness x]
       [--skin-saturation x]
  --list   only print the available presets/outfits/hair/poses/skins for the gender, then stop.
  --reset-body  zero the preset's body proportions, height and ageing keys (adult offsets; use for children)
  --brows none|name  hide HG's strand eyebrows (photo faces bring their own) or keep one system
  --iris "#RRGGBB"  eye colour;  --height-cm 165  body height
  --sclera "#C9BBB0"  eye-white colour;  --lash-dark 0.02  lash lightness;  --lash-length 0.75  lash length factor
  --fixed-camera  same framing for everyone (0.95 m target, 3.6 m away): for lineups
  --face-scale x  multiply the preset's face sliders (0.5 softens adult features for a child)
  --gown "#RRGGBB" [--gown-flare 1.0]  make the outfit's dress a floor-length flared gown in that colour
  --key    set any shape key or livekey, e.g. --key child_6=1 (the child livekeys from humgen/child/)
  N        an index into that list; a name is matched as a substring; omitted = HG's own default.

Writes <out_dir>/human.blend, front/side/three_quarter/face PNGs (Eevee), and human.json (what was chosen,
any errors). Prints MAKE_HUMAN_OK at the end (Blender exits 0 even when a script fails).
"""
import json
import math
import os
import sys

import bpy
from mathutils import Vector

from HumGen3D import Human

argv = sys.argv[sys.argv.index("--") + 1:]
out = os.path.abspath(argv[0])
opts = {"--gender": "female"}
flags = set()
keys_to_set = []  # (name, value) from --key name=value, in order
i = 1
while i < len(argv):
    if argv[i] in ("--list", "--reset-body", "--child-look", "--fixed-camera"):
        flags.add(argv[i][2:]); i += 1
    elif argv[i] == "--key":
        name, value = argv[i + 1].rsplit("=", 1)
        keys_to_set.append((name, float(value))); i += 2
    else:
        opts[argv[i]] = argv[i + 1]; i += 2
os.makedirs(out, exist_ok=True)
gender = opts["--gender"]
report = {"gender": gender, "errors": []}


def pick(options, choice):
    """Choice is an index, a substring of the option path, or None (keep default)."""
    if choice is None:
        return None
    if choice.isdigit():
        return options[int(choice)]
    hits = [o for o in options if choice.lower() in o.lower()]
    if not hits:
        raise SystemExit(f"MAKE_HUMAN_FAILED no option matches {choice!r}; options: {options}")
    return hits[0]


presets = Human.get_preset_options(gender)
if "list" in flags:
    print("PRESETS", json.dumps(presets, indent=1))
preset = pick(presets, opts.get("--preset")) or presets[0]

# start from an empty scene
for ob in list(bpy.data.objects):
    bpy.data.objects.remove(ob, do_unlink=True)

human = Human.from_preset(preset)
report["preset"] = preset
print("HUMAN", human.name, preset)

if "list" in flags:
    for kind, setting in (("OUTFITS", human.clothing.outfit), ("FOOTWEAR", human.clothing.footwear),
                          ("HAIR", human.hair.regular_hair), ("POSES", human.pose),
                          ("SKINS", human.skin.texture)):
        print(kind, json.dumps(setting.get_options(), indent=1))
    print("MAKE_HUMAN_OK")
    raise SystemExit

# body shape first, so clothes and hair are fitted to the final body
if "reset-body" in flags or keys_to_set or "--face-scale" in opts:
    if "reset-body" in flags:  # the preset's body proportions, height and ageing are adult offsets: drop them
        for key in human.keys.all_livekeys:
            if key.category == "body_proportions" or key.name in ("height_150", "height_200") \
                    or key.name.startswith("aged_"):
                if key.value:
                    key.set_without_update(0.0)
        report["reset_body"] = True
    if "--face-scale" in opts:  # the preset's face sliders are adult features: soften them for a child
        fs = float(opts["--face-scale"])
        for key in human.keys.all_livekeys:
            if key.category in ("face_proportions", "face_presets") and key.value:
                key.set_without_update(key.value * fs)
        report["face_scale"] = fs
    for name, value in keys_to_set:
        key = human.keys.get(name)
        if key is None:
            raise SystemExit(f"MAKE_HUMAN_FAILED no key {name!r}")
        key.set_without_update(value) if hasattr(key, "set_without_update") else setattr(key, "value", value)
    report["keys"] = dict(keys_to_set)
    human.keys.update_human_from_key_change(bpy.context)
if "--skin" in opts:
    choice = pick(human.skin.texture.get_options(), opts["--skin"])
    human.skin.texture.set(choice)
    report["skin"] = choice
if "child-look" in flags:  # adult presets carry ageing, makeup and strong pore detail: a child has none
    human.age.age_color.value = 0.0
    human.age.age_wrinkles.value = 0.0
    sk = human.skin
    sk.cavity_strength.value = 0.0
    sk.normal_strength.value = 0.6
    sk.freckles.value = 0.0
    sk.splotches.value = 0.0
    gs = getattr(sk, "gender_specific", None)
    for name in ("foundation_amount", "blush_opacity", "lipstick_opacity", "eyeliner_opacity", "eyebrows_opacity"):
        if gs is not None and hasattr(gs, name):
            getattr(gs, name).value = 0.0
    report["child_look"] = True
for name in ("tone", "redness", "saturation"):
    if f"--skin-{name}" in opts:
        getattr(human.skin, name).value = float(opts[f"--skin-{name}"])
        report[f"skin_{name}"] = float(opts[f"--skin-{name}"])

for flag, setting in (("--outfit", human.clothing.outfit), ("--footwear", human.clothing.footwear),
                      ("--hair", human.hair.regular_hair),
                      ("--pose", human.pose)):
    choice = pick(setting.get_options(), opts.get(flag))
    if choice:
        try:
            setting.set(choice)
            report[flag[2:]] = choice
        except Exception as e:  # noqa: BLE001
            report["errors"].append(f"{flag} {choice}: {e!r}")
if "child-look" in flags:  # after the hairstyle is loaded: dark, no grey; no beard or beard shadow
    if gender == "male":
        try:
            human.hair.face_hair.remove_all()
        except Exception as e:  # noqa: BLE001
            report["errors"].append(f"face hair: {e!r}")
        gs = human.skin.gender_specific
        for name in ("mustache_shadow", "beard_shadow"):
            if hasattr(gs, name):
                getattr(gs, name).value = 0.0
    hair = human.hair.regular_hair
    hair.lightness.value, hair.redness.value = 0.05, 0.6
    hair.salt_and_pepper.value, hair.roots.value = 0.0, 0.0
if "--gown" in opts:  # the outfit's dress made floor-length and flared (gown.py)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from gown import make_gown

    hx = opts["--gown"].lstrip("#")
    srgb = [int(hx[k:k + 2], 16) / 255 for k in (0, 2, 4)]
    lin = tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb)
    make_gown(human, color=lin, hem_flare=float(opts.get("--gown-flare", "1.0")))
    report["gown"] = opts["--gown"]
if "--brows" in opts:  # "none": hide HG's strand brows (a photo face brings its own); else a system's name
    for m in human.objects.body.modifiers:
        if m.type == "PARTICLE_SYSTEM" and m.particle_system.name.startswith("Eyebrows"):
            show = opts["--brows"] != "none" and opts["--brows"] in m.particle_system.name
            m.show_viewport = m.show_render = show
    report["brows"] = opts["--brows"]
if "--iris" in opts:
    hx = opts["--iris"].lstrip("#")
    srgb = [int(hx[k:k + 2], 16) / 255 for k in (0, 2, 4)]
    human.eyes.iris_color.value = (*[c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in srgb], 1)
    report["iris"] = opts["--iris"]
def _lin(hexstr):
    hx = hexstr.lstrip("#")
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4
                 for c in (int(hx[k:k + 2], 16) / 255 for k in (0, 2, 4)))


if "--sclera" in opts:  # HG's whites are near pure white: real eye whites are dimmer and warm
    human.eyes.sclera_color.value = (*_lin(opts["--sclera"]), 1)
    report["sclera"] = opts["--sclera"]
if "--lash-dark" in opts or "--lash-length" in opts:
    lashes = human.hair.eyelashes
    if "--lash-dark" in opts:
        lashes.lightness.value = float(opts["--lash-dark"])
        lashes.redness.value = 0.3
    if "--lash-length" in opts:
        f = float(opts["--lash-length"])
        for m in human.objects.body.modifiers:
            if m.type == "PARTICLE_SYSTEM" and m.particle_system.name.startswith("Eyelash"):
                st = m.particle_system.settings  # HG lashes: edited guide strands with children
                st.child_length = st.child_length * f
                st.radius_scale = st.radius_scale * 0.7  # thinner too
    report["lashes"] = {k: opts[k] for k in ("--lash-dark", "--lash-length") if k in opts}
if "--height-cm" in opts:
    human.height.set(float(opts["--height-cm"]))
    report["height_cm_set"] = float(opts["--height-cm"])
if "--age" in opts:
    human.age.set(int(opts["--age"]))
    report["age"] = int(opts["--age"])



def fix_eye_cornea():
    """Eevee renders HG's refracting, dithered cornea as speckle: blended transparency without refraction."""
    for mt in human.objects.eyes.data.materials:
        if mt and mt.name.startswith(".HG_Eyes_Outer"):
            mt.surface_render_method = "BLENDED"
            mt.use_raytrace_refraction = False


fix_eye_cornea()
# --- check renders ---
scene = bpy.context.scene
scene.render.engine = "BLENDER_EEVEE"
scene.render.resolution_x, scene.render.resolution_y = 900, 1200
scene.render.film_transparent = False
world = bpy.data.worlds.new("check_world")
world.use_nodes = True
world.node_tree.nodes["Background"].inputs[0].default_value = (0.35, 0.35, 0.37, 1)
world.node_tree.nodes["Background"].inputs[1].default_value = 0.6
scene.world = world


def light(name, loc, energy):
    d = bpy.data.lights.new(name, "AREA")
    d.energy, d.size = energy, 3
    d.shape = "DISK"  # round catchlights in the eyes
    o = bpy.data.objects.new(name, d)
    scene.collection.objects.link(o)
    o.location = loc
    o.rotation_euler = (Vector((0, 0, 1)) - Vector(loc)).to_track_quat("-Z", "Y").to_euler()


light("key", (2.5, -3, 3), 800)
light("fill", (-3, -2, 2), 250)
light("rim", (0, 3, 3), 400)

cam_data = bpy.data.cameras.new("check_cam")
cam = bpy.data.objects.new("check_cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam

body = human.objects.body
bpy.context.view_layer.update()
pts = [body.matrix_world @ Vector(c) for c in body.bound_box]
zmin, zmax = min(p.z for p in pts), max(p.z for p in pts)
height = zmax - zmin
report["height_m"] = round(height, 3)


def shoot(name, yaw_deg, target_z, dist, lens):
    yaw = math.radians(yaw_deg)
    cam_data.lens = lens
    cam.location = (dist * math.sin(yaw), -dist * math.cos(yaw), target_z)
    cam.rotation_euler = (Vector((0, 0, target_z)) - cam.location).to_track_quat("-Z", "Y").to_euler()
    scene.render.filepath = os.path.join(out, name + ".png")
    bpy.ops.render.render(write_still=True)
    print("RENDERED", scene.render.filepath)


mid = zmin + height / 2
if "fixed-camera" in flags:  # same framing for every human (age lineups): front and three-quarter only
    shoot("front", 0, 0.95, 3.6, 50)
    shoot("three_quarter", 35, 0.95, 3.6, 50)
else:
    shoot("front", 0, mid, 3.4, 50)
    shoot("three_quarter", 35, mid, 3.4, 50)
    shoot("side", 90, mid, 3.4, 50)
    shoot("face", 15, zmax - 0.13, 1.4, 85)

bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out, "human.blend"))
with open(os.path.join(out, "human.json"), "w") as f:
    json.dump(report, f, indent=2)
print("REPORT", json.dumps(report))
print("MAKE_HUMAN_OK")
