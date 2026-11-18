"""Turn a Human Generator dress into a floor-length gown (used by make_human.py --gown).

The garment's current shape (its shape keys mixed, so a child's refit is kept) becomes its base. Below the
hips the skirt is flared into a bell; the hem is then extended with new rings of mesh down to just above the
floor, flaring on out. Skirt points are weighted to the hips bone (spine), blending in from the hip line, so leg
poses don't drag the skirt. The fabric gets a colour and a satin look (smoother, shinier, weave turned down).
"""
import math

import bmesh
import bpy
import numpy as np
from mathutils import Vector


def _smoothstep(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def make_gown(human, color=(0.55, 0.05, 0.25), hem_flare=1.0, ring_step=0.025, floor_gap=0.012):
    """human: an HG Human with a dress-like outfit on. Returns the gown object."""
    rig = human.objects.rig
    cloth = max(human.clothing.outfit.objects, key=lambda o: len(o.data.vertices))
    ctx = bpy.context
    # 1. bake the shape keys (the child refit) into the mesh
    for o in ctx.selected_objects:
        o.select_set(False)
    cloth.hide_set(False)
    cloth.select_set(True)
    ctx.view_layer.objects.active = cloth
    if cloth.data.shape_keys:
        bpy.ops.object.shape_key_remove(all=True, apply_mix=True)
    # HG garments are stored rotated (local y is up): put the transform into the mesh so local = world, z up
    cloth.data.transform(cloth.matrix_world)
    cloth.matrix_parent_inverse = rig.matrix_world.inverted()
    cloth.matrix_basis.identity()
    ctx.view_layer.update()

    # 2. landmarks from the rig's rest pose, in the cloth's local space
    to_local = cloth.matrix_world.inverted() @ rig.matrix_world
    bones = rig.data.bones
    hip_l, hip_r = to_local @ bones["thigh.L"].head_local, to_local @ bones["thigh.R"].head_local
    knee_z = (to_local @ bones["shin.L"].head_local).z
    centre = (hip_l + hip_r) / 2
    z_hip = centre.z + 0.04  # flare starts a little above the hip joints
    floor_z = (cloth.matrix_world.inverted() @ Vector((0, 0, 0))).z + floor_gap
    half_hips = abs(hip_l.x - hip_r.x) / 2

    me = cloth.data
    # bell profile: extra radius by depth below the hip line (0 at the hips, hem_flare*legs-length/2.2 at floor)
    depth_total = z_hip - floor_z
    max_extra = hem_flare * depth_total / 2.2

    def growth(z):
        d = np.clip((z_hip - z) / depth_total, 0.0, 1.0)
        return max_extra * d ** 1.25 * _smoothstep(d / 0.2)

    # 3. the skirt is rebuilt from the hips down (the dress's own hem is a folded double layer that shows
    # as a step if extended): cut the garment at the hip line, then add even rings down to the floor
    bm = bmesh.new()
    bm.from_mesh(me)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.co.z < z_hip], context="VERTS")
    boundary = [e for e in bm.edges if e.is_boundary]
    loops, seen = [], set()
    for e in boundary:
        if e in seen:
            continue
        loop, stack = set(), [e]
        while stack:
            x = stack.pop()
            if x in loop:
                continue
            loop.add(x)
            for v in x.verts:
                stack.extend(y for y in v.link_edges if y.is_boundary and y not in loop)
        seen |= loop
        loops.append(loop)
    cut = min(loops, key=lambda lp: np.mean([v.co.z for e in lp for v in e.verts]))
    cut_verts = {v for e in cut for v in e.verts}
    c2 = np.array([centre.x, centre.y])
    base = {v: (np.array(v.co[:2]) - c2) for v in cut_verts}  # each column's direction and hip radius
    n_rings = max(1, int(math.ceil((z_hip - floor_z) / ring_step)))
    edges, prev = list(cut), {v: v for v in cut_verts}
    for k in range(1, n_rings + 1):
        res = bmesh.ops.extrude_edge_only(bm, edges=edges)
        nv = [g for g in res["geom"] if isinstance(g, bmesh.types.BMVert)]
        ne = [g for g in res["geom"] if isinstance(g, bmesh.types.BMEdge)]
        z = z_hip + (floor_z - z_hip) * k / n_rings
        g = float(growth(z))
        for v in nv:
            # the column this new point continues: the cut point it was extruded from (nearest in xy)
            src = min(prev.items(), key=lambda kv: (kv[0].co - v.co).length_squared)[1]
            b = base[src]
            r = np.linalg.norm(b) + 1e-9
            xy = c2 + b / r * (r + g)
            v.co = Vector((xy[0], xy[1], z))
            base[v] = b
        prev = {v: (v if v in base else v) for v in nv}
        edges = [e for e in ne if all(v in nv for v in e.verts)]
    hem_z = z_hip
    # soft folds: a wave around the skirt, growing toward the hem
    for v in bm.verts:
        d = (z_hip - v.co.z) / depth_total
        if d <= 0.15:
            continue
        amp = 0.018 * hem_flare * ((min(d, 1.0) - 0.15) / 0.85) ** 1.5
        rel = Vector((v.co.x - centre.x, v.co.y - centre.y, 0))
        if rel.length < 1e-6:
            continue
        theta = math.atan2(rel.y, rel.x)
        wave = math.sin(11 * theta + 0.6 * math.sin(3 * theta))
        rel_n = rel.normalized()
        v.co.x += rel_n.x * amp * wave
        v.co.y += rel_n.y * amp * wave
    for f in bm.faces:  # one smooth surface: the garment's own sharp edges and creases made a ring at the old hem
        f.smooth = True
    for e in bm.edges:
        e.smooth = True
    bm.normal_update()
    bm.to_mesh(me)
    bm.free()
    if me.has_custom_normals:  # stored normals from the original garment no longer match the new shape
        ctx.view_layer.objects.active = cloth
        bpy.ops.mesh.customdata_custom_splitnormals_clear()
    for name in ("crease_edge", "crease_vert", "sharp_edge", "sharp_face"):
        if name in me.attributes:
            me.attributes.remove(me.attributes[name])
    me.update()

    # 4. weights: skirt follows the hips bone, blending in below the hip line, fully below the knees' top
    spine = cloth.vertex_groups.get("spine") or cloth.vertex_groups.new(name="spine")
    leg_groups = {g.index for g in cloth.vertex_groups if g.name.split(".")[0] in ("thigh", "shin", "foot", "toe")}
    for v in me.vertices:
        t = float(_smoothstep(np.array((z_hip - v.co.z) / max(z_hip - knee_z, 1e-3))))
        if t <= 0:
            continue
        moved = 0.0  # weight taken off the legs goes to the hips, so the total stays the same
        for g in v.groups:
            if g.group in leg_groups:
                moved += g.weight * t
                g.weight *= (1 - t)
        cur = next((g.weight for g in v.groups if g.group == spine.index), 0.0)
        spine.add([v.index], min(1.0, cur + moved), "REPLACE")
    # below the knees (incl. the new rings): fully on the hips
    for v in me.vertices:
        if v.co.z < knee_z:
            for g in v.groups:
                if g.group != spine.index:
                    g.weight = 0.0
            spine.add([v.index], 1.0, "REPLACE")

    # 5. satin look and colour
    # its own material: the knit texture stretches on the new rings and fights the colour
    mat = bpy.data.materials.new("HG_Gown_Satin")
    nt = mat.node_tree
    bsdf = nt.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = 0.42
    for name, val in (("Sheen Weight", 0.3), ("Sheen Roughness", 0.35), ("Specular IOR Level", 0.3)):
        if name in bsdf.inputs:
            bsdf.inputs[name].default_value = val
    if "Sheen Tint" in bsdf.inputs:
        bsdf.inputs["Sheen Tint"].default_value = (*[min(1.0, c * 1.6 + 0.15) for c in color], 1.0)
    # a faint woven grain so it isn't plastic
    tex = nt.nodes.new("ShaderNodeTexNoise")
    tex.inputs["Scale"].default_value = 900.0
    bump = nt.nodes.new("ShaderNodeBump")
    bump.inputs["Strength"].default_value = 0.04
    nt.links.new(tex.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
    me.materials.clear()
    me.materials.append(mat)
    for m in cloth.modifiers:
        if m.type == "SUBSURF":
            break
    else:
        sub = cloth.modifiers.new("gown_smooth", "SUBSURF")
        sub.levels = sub.render_levels = 1
    cloth.name = "HG_Gown"
    print("GOWN", cloth.name, "rings", n_rings, "verts", len(me.vertices), f"hem {hem_z:.3f} -> {floor_z:.3f}")
    return cloth
