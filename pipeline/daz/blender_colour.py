"""Skin and hair colour of the imported Daz character matched to the portrait in Blender (used by blender_daz_render.py).

A straight-on EEVEE render; landmarks2d.py measures the cheeks (skin) and the hair above the forehead on it and on the
portrait; each round multiplies the skin materials' colour (Head, Body, Arms, Legs, ...) and the hair materials' colour
by (portrait / render)^0.8 per channel in linear light (damped, clamped). The gains go into a "Colour match" multiply
node in each material, so the textures are untouched and the gains can be read back (returned and printed).
"""
import os

import bpy
import numpy as np
from mathutils import Vector

import blender_likeness as BL

SKIN_MATS = ("Head", "Body", "Arms", "Legs", "Fingernails")
COLOUR_INPUTS = ("Base Color", "Color", "Diffuse Color", "Subsurface Color", "Tip Color", "Root Color")


def lin(c):
    c = np.asarray(c, float) / 255
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def set_gain(mat, gain):
    """Multiply every colour input of the material's shaders by gain (one 'Colour match' node per input)."""
    nt = mat.node_tree
    if nt is None:
        return 0
    n = 0
    for node in list(nt.nodes):
        if node.type in ("GROUP_INPUT", "GROUP_OUTPUT", "OUTPUT_MATERIAL") or node.name.startswith("Colour match"):
            continue
        if not (node.type.startswith("BSDF") or node.type in ("SUBSURFACE_SCATTERING", "GROUP")):
            continue
        for sock in node.inputs:
            if sock.name not in COLOUR_INPUTS or sock.type != "RGBA":
                continue
            key = f"Colour match {node.name} {sock.name}"
            mix = nt.nodes.get(key)
            if mix is None:
                mix = nt.nodes.new("ShaderNodeMix")
                mix.name = key
                mix.data_type, mix.blend_type = "RGBA", "MULTIPLY"
                mix.inputs["Factor"].default_value = 1.0
                a_in, b_in = mix.inputs[6], mix.inputs[7]
                if sock.is_linked:
                    nt.links.new(sock.links[0].from_socket, a_in)
                else:
                    a_in.default_value = sock.default_value
                nt.links.new(mix.outputs[2], sock)
            mix.inputs[7].default_value = (*[float(g) for g in gain], 1.0)
            n += 1
    return n


def run(photo, eye, work, rounds=3, res=(480, 600)):
    sc = bpy.context.scene
    skin = [m for m in bpy.data.materials if m.name.split(".")[0].split("-")[0] in SKIN_MATS]
    hair_objs = [o for o in sc.objects if o.type == "MESH" and "hair" in o.name.lower()
                 and not any(k in o.name for k in BL.FOLLOW)]
    hair = list({m for o in hair_objs for m in o.data.materials if m})
    cam = bpy.data.objects.new("col_cam", bpy.data.cameras.new("col_cam"))
    sc.collection.objects.link(cam)
    cam.data.lens = 85
    tgt = eye + Vector((0, 0, -0.01))
    cam.location = tgt + Vector((0, -0.85, 0))
    cam.rotation_euler = (tgt - cam.location).to_track_quat("-Z", "Y").to_euler()
    old = (sc.camera, sc.render.resolution_x, sc.render.resolution_y, sc.render.filepath)
    sc.camera = cam
    sc.render.resolution_x, sc.render.resolution_y = res
    BL.landmarks([photo], work)
    want = BL.landmarks.colours.get(photo)
    g_skin, g_hair = np.ones(3), np.ones(3)
    prev_ratio, prev_gain = None, None
    log = []
    print("COLOUR materials: skin %s (%d colour inputs) | hair %s (%d)" % (
        [m.name for m in skin], sum(set_gain(m, g_skin) for m in skin), [m.name for m in hair],
        sum(set_gain(m, g_hair) for m in hair)), flush=True)
    for rnd in range(rounds + 1):
        for m in skin:
            set_gain(m, g_skin)
        for m in hair:
            set_gain(m, g_hair)
        img = os.path.join(work, f"colour_{rnd}.png")
        sc.render.filepath = img
        bpy.ops.render.render(write_still=True)
        BL.landmarks([img], work)
        have = BL.landmarks.colours.get(img)
        if not want or not have:
            print("COLOUR no face found, colours left as they are")
            break
        print("COLOUR round %d: skin portrait %s render %s | hair portrait %s render %s | gains skin %s hair %s" % (
            rnd, [round(v) for v in want["skin"]], [round(v) for v in have["skin"]], [round(v) for v in want["hair"]],
            [round(v) for v in have["hair"]], g_skin.round(3).tolist(), g_hair.round(3).tolist()), flush=True)
        log.append({"round": rnd, "skin": have["skin"], "hair": have["hair"], "gain_skin": g_skin.tolist(), "gain_hair": g_hair.tolist()})
        if rnd == rounds:
            break
        g_skin = np.clip(g_skin * np.clip((lin(want["skin"]) / np.maximum(lin(have["skin"]), 1e-4)) ** 0.8, 0.6, 1.6), 0.2, 4)
        if hair:  # hair answers the multiplier very unevenly: half steps, and halve again where a channel overshoots
            ratio = lin(want["hair"]) / np.maximum(lin(have["hair"]), 1e-4)
            if rnd > 0 and prev_ratio is not None:
                flip = np.sign(np.log(ratio)) != np.sign(np.log(prev_ratio))
                g_new = g_hair * np.clip(ratio ** 0.5, 0.7, 1.4)
                g_new[flip] = np.sqrt(g_hair[flip] * prev_gain[flip])  # between the last two: bisection in log
            else:
                g_new = g_hair * np.clip(ratio ** 0.5, 0.7, 1.4)
            prev_ratio, prev_gain = ratio, g_hair.copy()
            g_hair = np.clip(g_new, 0.05, 4)
    sc.camera, sc.render.resolution_x, sc.render.resolution_y, sc.render.filepath = old
    bpy.data.objects.remove(cam)
    return {"skin_gain_linear_rgb": g_skin.tolist(), "hair_gain_linear_rgb": g_hair.tolist(), "rounds": log}
