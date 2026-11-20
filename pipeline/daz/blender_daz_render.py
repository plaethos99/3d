"""Blender side of daz_to_blender.py: Easy Import of a Daz scene (Diffeomorphic), studio lights, three renders.

blender -b --factory-startup -P blender_daz_render.py -- <scene.duf> <root_paths.json> <out_dir> <name>
       [--engine eevee|cycles] [--samples 64] [--photo portrait.png --morph <Daz morph .dsf>] [--likeness 3]
--photo: the face matched to the portrait first (blender_likeness.py), and written into the Daz morph with --morph.
Shots like g9_build.dsa (same framing, so Daz and Blender renders compare side by side): face (front, 85 mm),
face34 (35 degrees), body (25 degrees, 50 mm). Prints BLENDER_DAZ_RENDER_OK.
"""
import math
import os
import sys
import time

import addon_utils
import bpy
from mathutils import Vector

argv = sys.argv[sys.argv.index("--") + 1:]
scene_path, roots, out_dir, name = argv[:4]


def opt(k, default):
    return argv[argv.index(k) + 1] if k in argv else default


engine, samples = opt("--engine", "eevee"), int(opt("--samples", "64"))
addon_utils.enable("bl_ext.user_default.import_daz", default_set=True)
for o in list(bpy.data.objects):
    bpy.data.objects.remove(o)

bpy.ops.daz.load_root_paths(filepath=roots)
try:
    bpy.ops.daz.set_silent_mode()
except Exception:
    pass
t0 = time.time()
res = bpy.ops.daz.easy_import_daz(filepath=scene_path, directory=os.path.dirname(scene_path) + os.sep,
                                   files=[{"name": os.path.basename(scene_path)}],
                                   useMergeRigs=True, useEliminateEmpties=True)
print("IMPORTED", res, f"{time.time() - t0:.0f}s", len(bpy.data.objects), "objects:",
      ", ".join(o.name for o in bpy.data.objects if o.type in ("ARMATURE", "MESH"))[:300], flush=True)
rig = next((o for o in bpy.data.objects if o.type == "ARMATURE"), None)
if rig is None:
    raise SystemExit("WARN no armature imported")
bpy.context.view_layer.update()


def bone_ws(names):
    for n in names:
        b = rig.pose.bones.get(n)
        if b:
            return rig.matrix_world @ b.head
    return None


le, re_ = bone_ws(["l_eye", "lEye"]), bone_ws(["r_eye", "rEye"])
if le is None or re_ is None:
    meshes = [o for o in bpy.data.objects if o.type == "MESH"]
    zs = [(o.matrix_world @ Vector(c)).z for o in meshes for c in o.bound_box]
    eye = Vector((0, 0, max(zs) * 0.93))
    print("WARN no eye bones, eyes guessed at", eye)
else:
    eye = (le + re_) / 2
print("CAMERA eyes at", [round(v, 3) for v in eye], flush=True)

sc = bpy.context.scene
sc.render.engine = "CYCLES" if engine == "cycles" else "BLENDER_EEVEE"
if engine == "cycles":
    sc.cycles.samples = samples
    sc.cycles.use_denoising = True
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        for d in prefs.devices:
            d.use = True
        sc.cycles.device = "GPU"
    except Exception as e:
        print("WARN cycles GPU", e)
else:
    try:
        sc.eevee.taa_render_samples = samples
        sc.eevee.use_raytracing = True
        sc.eevee.use_shadows = True
    except Exception:
        pass
sc.render.film_transparent = True
sc.view_settings.view_transform = "AgX"
sc.view_settings.look = "AgX - Medium High Contrast"
world = bpy.data.worlds.new("studio") if not sc.world else sc.world
sc.world = world
world.use_nodes = True
bg = world.node_tree.nodes.get("Background")
bg.inputs[0].default_value = (0.8, 0.8, 0.82, 1)
bg.inputs[1].default_value = 0.25


def light(nm, energy, size, az, el, dist, color=(1, 1, 1)):
    d = bpy.data.lights.new(nm, "AREA")
    d.energy, d.size, d.color = energy, size, color
    o = bpy.data.objects.new(nm, d)
    sc.collection.objects.link(o)
    a, e = math.radians(az), math.radians(el)
    o.location = eye + Vector((math.sin(a) * math.cos(e), -math.cos(a) * math.cos(e), math.sin(e))) * dist
    o.rotation_euler = (eye - o.location).to_track_quat("-Z", "Y").to_euler()


light("key", 140, 1.2, -35, 25, 2.0, (1, 0.97, 0.93))
light("fill", 50, 1.5, 45, 5, 2.2, (0.93, 0.96, 1))
light("rim", 90, 0.8, 160, 35, 2.0)

cam_d = bpy.data.cameras.new("cam")
cam = bpy.data.objects.new("cam", cam_d)
sc.collection.objects.link(cam)
sc.camera = cam


def shot(tag, az, dist, lens, target, w, h):
    a = math.radians(az)
    cam_d.lens = lens
    cam.location = target + Vector((math.sin(a), -math.cos(a), 0)) * dist
    cam.rotation_euler = (target - cam.location).to_track_quat("-Z", "Y").to_euler()
    sc.render.resolution_x, sc.render.resolution_y = w, h
    sc.render.filepath = os.path.join(out_dir, f"{name}_bl_{tag}.png")
    t = time.time()
    bpy.ops.render.render(write_still=True)
    print("RENDER", tag, os.path.exists(sc.render.filepath), f"{time.time() - t:.0f}s", flush=True)


# the face matched to the portrait (and written back into the Daz morph), before the final pictures
if opt("--photo", ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import blender_likeness  # noqa: E402
    body = next(o for o in bpy.data.objects if o.type == "MESH" and o.name.startswith("Genesis 9 Mesh"))
    blender_likeness.run(opt("--photo", ""), body, eye, out_dir, morph_path=opt("--morph", "") or None,
                         rounds=int(opt("--likeness", "3")))
    # skin and hair colour matched to the portrait (multiply nodes in the materials; saved in the .blend)
    import json as _json  # noqa: E402
    import blender_colour  # noqa: E402
    if int(opt("--colour", "0")):  # off by default: the skin textures are Daz's (calibrated with Iray)
        col = blender_colour.run(opt("--photo", ""), eye, out_dir, rounds=int(opt("--colour", "0")))
        _json.dump(col, open(os.path.join(out_dir, f"{name}_bl_colour.json"), "w"), indent=1)

# head hair grown in Blender (blender_hair.py)
if opt("--hair", ""):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import json as _json2  # noqa: E402
    import blender_hair  # noqa: E402
    blender_hair.grow(rig, eye, _json2.load(open(opt("--hair", ""))))

# Daz framing in metres (g9_build.dsa: face 75 cm at 85 mm, body 380 cm at 50 mm, body target at (eye+5 cm)*0.55)
face_t = eye + Vector((0, 0, -0.04))
shot("face", 0, 0.75, 85, face_t, 600, 750)
shot("face34", 35, 0.75, 85, face_t, 600, 750)
shot("body", 25, 3.8, 50, Vector((eye.x, eye.y, (eye.z + 0.05) * 0.55)), 560, 800)
bpy.ops.wm.save_as_mainfile(filepath=os.path.join(out_dir, f"{name}_bl.blend"))
print("BLENDER_DAZ_RENDER_OK")
