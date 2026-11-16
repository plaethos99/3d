"""Give rigged garments real folds with Blender's cloth simulation, then keep the result as their shape.

Usage: blender -b character.blend -P simulate_cloth.py -- <out.blend> <garments.json>

garments.json: a list of garments, each
  {"objects": ["leg_L"], "subdivide": 1, "inner_mm": 15, "grow": 0.03, "frames": 40,
   "pin_above": 0.62, "pin_fade_mm": 70, "pin_near": ["hem"], "pin_near_mm": 15,
   "also_collide": ["shoe_L"], "bending": 0.5, "stiffness": 15}

How each garment is handled, like a cloth artist would:
  1. Its own shape, pulled in by inner_mm along the surface, becomes the body it rests on
     (the clothes were built without a body underneath), plus any other colliders.
  2. It's pinned where it's held: above a height (waistband, shoulders) and next to the parts it's
     sewn to (cuffs, hem, front band), so it can't come apart from them.
  3. The fabric is made slightly larger than its pattern (grow), like real clothes cut with ease.
     Held at its pins, the extra fabric can't go anywhere but into folds.
  4. The simulation runs in the rest pose for some frames, and the settled shape replaces the old
     one. Skin weights are kept, so the garment still follows the skeleton.
"""
import json
import sys

import bpy
import numpy as np
from mathutils.bvhtree import BVHTree

out, config_path = sys.argv[sys.argv.index("--") + 1:][:2]
with open(config_path, encoding="utf-8") as f:
    garments = json.load(f)
scene = bpy.context.scene
rig = bpy.data.objects.get("joyce_rig")
if rig is not None:
    rig.data.pose_position = "REST"


def select_only(obj):
    for other in bpy.context.view_layer.objects:
        other.select_set(False)
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj


def world_points(obj):
    return np.array([obj.matrix_world @ v.co for v in obj.data.vertices])


report = {}
for g in garments:
    for name in g["objects"]:
        obj = bpy.data.objects[name]
        select_only(obj)
        # Finer mesh, so it can hold folds.
        if g.get("subdivide", 0):
            sub = obj.modifiers.new("cloth_subdivide", "SUBSURF")
            sub.levels = g["subdivide"]
            bpy.ops.object.modifier_move_to_index(modifier=sub.name, index=0)
            bpy.ops.object.modifier_apply(modifier=sub.name)
        co = world_points(obj)
        normals = np.array([obj.matrix_world.to_3x3() @ v.normal for v in obj.data.vertices])

        # 1. A body underneath: the garment's own shape, pulled in.
        proxy_mesh = obj.data.copy()
        proxy = bpy.data.objects.new(f"{name}_body", proxy_mesh)
        scene.collection.objects.link(proxy)
        proxy.matrix_world = obj.matrix_world.copy()
        inner = g.get("inner_mm", 15) / 1000
        local_n = np.array([v.normal for v in proxy_mesh.vertices])
        local_co = np.array([v.co for v in proxy_mesh.vertices])
        # Normals of open garments can face in or out; pull toward the garment's own middle line.
        middle = local_co.mean(axis=0)
        flip = np.sign(((local_co - middle) * local_n).sum(axis=1))[:, None]
        proxy_mesh.vertices.foreach_set("co", (local_co - local_n * flip * inner).ravel())
        proxy_mesh.update()
        proxy.modifiers.clear()
        proxy.vertex_groups.clear()
        proxy.parent = None
        proxy.hide_render = True
        proxy.modifiers.new("collision", "COLLISION")
        proxy.collision.thickness_outer = 0.004
        for other in g.get("also_collide", []):
            o = bpy.data.objects[other]
            if not any(m.type == "COLLISION" for m in o.modifiers):
                o.modifiers.new("collision", "COLLISION")
                o.collision.thickness_outer = 0.003

        # 2. Pins: above a height, fading out below it, and next to sewn-on parts.
        weight = np.zeros(len(co))
        if "pin_above" in g:
            fade = g.get("pin_fade_mm", 60) / 1000
            weight = np.maximum(weight, np.clip((co[:, 2] - (g["pin_above"] - fade)) / fade, 0, 1))
        if "pin_below" in g:
            fade = g.get("pin_fade_mm", 60) / 1000
            weight = np.maximum(weight, np.clip(((g["pin_below"] + fade) - co[:, 2]) / fade, 0, 1))
        near = g.get("pin_near_mm", 15) / 1000
        for other in g.get("pin_near", []):
            o = bpy.data.objects[other]
            depsgraph = bpy.context.evaluated_depsgraph_get()
            tree = BVHTree.FromObject(o, depsgraph)
            for i, p in enumerate(co):
                hit = tree.find_nearest(o.matrix_world.inverted() @ __import__("mathutils").Vector(p))
                if hit[0] is not None:
                    d = (o.matrix_world @ hit[0] - __import__("mathutils").Vector(p)).length
                    weight[i] = max(weight[i], float(np.clip(1 - (d - near) / near, 0, 1)))
        pin = obj.vertex_groups.get("cloth_pin") or obj.vertex_groups.new(name="cloth_pin")
        for i, w in enumerate(weight):
            if w > 0:
                pin.add([i], float(w), "REPLACE")

        # Extra length (like trousers cut a little long): the fabric's pattern is taller than the gap
        # between its pins, so it buckles across, into horizontal folds, not lengthwise ones.
        if g.get("lengthen", 0):
            if obj.data.shape_keys is None:
                obj.shape_key_add(name="Basis")
            pattern = obj.shape_key_add(name="pattern")
            top = g.get("pin_above", float(co[:, 2].max()))
            local = np.array([v.co for v in obj.data.vertices])
            stretched = local.copy()
            stretched[:, 2] = top - (top - local[:, 2]) * (1 + g["lengthen"])
            stretched[:, 2] = np.where(local[:, 2] > top, local[:, 2], stretched[:, 2])
            pattern.data.foreach_set("co", stretched.ravel())

        # 3 and 4. Cloth, first in the stack, in the rest pose.
        cloth = obj.modifiers.new("cloth", "CLOTH")
        bpy.ops.object.modifier_move_to_index(modifier=cloth.name, index=0)
        s = cloth.settings
        s.quality = 8
        s.mass = g.get("mass", 0.25)
        s.air_damping = 2.0
        s.tension_stiffness = s.compression_stiffness = g.get("stiffness", 15)
        s.shear_stiffness = g.get("stiffness", 15) * 0.5
        s.bending_stiffness = g.get("bending", 0.5)
        s.shrink_min = -g.get("grow", 0.03)
        s.vertex_group_mass = "cloth_pin"
        if g.get("lengthen", 0):
            s.rest_shape_key = obj.data.shape_keys.key_blocks["pattern"]
        s.pin_stiffness = 1.0
        cloth.collision_settings.use_collision = True
        cloth.collision_settings.distance_min = 0.003
        cloth.collision_settings.use_self_collision = g.get("self_collision", False)
        cloth.point_cache.frame_start = 1
        cloth.point_cache.frame_end = g.get("frames", 40)
        report[name] = {"vertices": len(co), "pinned": int((weight > 0.5).sum())}
        print("CLOTH_SETUP", name, report[name], flush=True)

    # Run the simulation for this garment's objects together.
    frames = g.get("frames", 40)
    for frame in range(1, frames + 1):
        scene.frame_set(frame)
    for name in g["objects"]:
        obj = bpy.data.objects[name]
        before = world_points(obj)
        select_only(obj)
        # Read the settled shape (the skeleton is in its rest pose, so only the cloth moved it),
        # then make it the mesh itself. The pattern key and the simulation are no longer needed.
        depsgraph = bpy.context.evaluated_depsgraph_get()
        settled = obj.evaluated_get(depsgraph).to_mesh()
        coords = np.empty(len(settled.vertices) * 3)
        settled.vertices.foreach_get("co", coords)
        obj.evaluated_get(depsgraph).to_mesh_clear()
        obj.modifiers.remove(obj.modifiers["cloth"])
        if obj.data.shape_keys is not None:
            obj.shape_key_clear()
        obj.data.vertices.foreach_set("co", coords)
        obj.data.update()
        after = world_points(obj)
        moved = np.linalg.norm(after - before, axis=1) * 1000
        report[name]["moved_mm_median"] = round(float(np.median(moved)), 1)
        report[name]["moved_mm_max"] = round(float(moved.max()), 1)
        bpy.data.objects.remove(bpy.data.objects[f"{name}_body"], do_unlink=True)
        print("CLOTH_DONE", name, report[name], flush=True)
    scene.frame_set(1)

for obj in bpy.data.objects:
    for m in list(obj.modifiers):
        if m.type == "COLLISION":
            obj.modifiers.remove(m)
if rig is not None:
    rig.data.pose_position = "POSE"
scene.frame_set(1)
bpy.ops.wm.save_as_mainfile(filepath=out)
print("SIMULATE_CLOTH_OK")
