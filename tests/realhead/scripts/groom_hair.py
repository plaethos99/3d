"""Grow and comb strand hair on a head, the way a groomer would, inside a given hair volume.

Usage: blender -b character.blend -P groom_hair.py -- <out.blend> [options as KEY=value]

Objects used: "head" (the scalp), an outer hair surface (VOLUME=hair_base) giving the hairstyle's
silhouette, and optionally an armature "joyce_rig" with a "head" bone the hair is attached to.

Each strand starts on the scalp, inside the hairline, then:
  1. rises off the scalp into the volume (to a random depth, so the hair has layers),
  2. is combed: swept back over the top, falling down at the sides and back,
  3. waves from side to side and in and out, in phase with the strands of its lock,
  4. ends where the volume ends, curling a little at the tip.
Strands gather into locks around guide strands, so the hair reads as clumps rather than fuzz.
Options (defaults in brackets): STRANDS (30000), GUIDES (900), POINTS (18), WAVE_MM (9), WAVE_LENGTH_MM
(70), CLUMP (0.7), ROOT_RADIUS_MM (0.6), MELANIN (0.15), HAIRLINE_FRONT_DEG (48), END_FRONT_DEG (70),
END_SIDE_DEG (102), END_BACK_DEG (118), MAX_LENGTH_MM (200), LIFT_MM (12), REMOVE (hair_base,hair_inner).
"""
import math
import sys

import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

args = sys.argv[sys.argv.index("--") + 1:]
out = args[0]
opt = dict(a.split("=", 1) for a in args[1:])
STRANDS = int(opt.get("STRANDS", 30000))
GUIDES = int(opt.get("GUIDES", 900))
POINTS = int(opt.get("POINTS", 18))
WAVE = float(opt.get("WAVE_MM", 9)) / 1000
WAVELENGTH = float(opt.get("WAVE_LENGTH_MM", 70)) / 1000
CLUMP = float(opt.get("CLUMP", 0.7))
ROOT_RADIUS = float(opt.get("ROOT_RADIUS_MM", 0.6)) / 1000
MELANIN = float(opt.get("MELANIN", 0.15))
FRONT_LIMIT = float(opt.get("HAIRLINE_FRONT_DEG", 48))
END_FRONT = float(opt.get("END_FRONT_DEG", 70))
END_SIDE = float(opt.get("END_SIDE_DEG", 102))
END_BACK = float(opt.get("END_BACK_DEG", 118))
MAX_LENGTH = float(opt.get("MAX_LENGTH_MM", 200)) / 1000
LIFT = float(opt.get("LIFT_MM", 12)) / 1000
FRONT_RISE = float(opt.get("FRONT_RISE", 1.5))
STRAY = float(opt.get("STRAY_MM", 1.5)) / 1000
LAYER = float(opt.get("LAYER_MM", 14)) / 1000
TIP_CURL = float(opt.get("TIP_CURL_MM", 6)) / 1000
TIP_SHARE = float(opt.get("TIP_SHARE", 0.2))
VOLUME = opt.get("VOLUME", "hair_base")
REMOVE = [n for n in opt.get("REMOVE", "hair_base,hair_inner").split(",") if n]
rng = np.random.default_rng(int(opt.get("SEED", 11)))

head = bpy.data.objects["head"]
volume = bpy.data.objects[VOLUME]
depsgraph = bpy.context.evaluated_depsgraph_get()
shell = BVHTree.FromObject(volume, depsgraph)
inner = bpy.data.objects.get("hair_inner")
inner_co = np.array([inner.matrix_world @ v.co for v in inner.data.vertices]) if inner else None
# The skull's centre: middle of the scalp cap if there is one, else of the volume.
ref = inner_co if inner_co is not None else np.array([volume.matrix_world @ v.co for v in volume.data.vertices])
centre = (ref.min(axis=0) + ref.max(axis=0)) / 2
centre[2] = ref.min(axis=0)[2] + (ref.max(axis=0)[2] - ref.min(axis=0)[2]) * 0.45
print("SKULL_CENTRE", np.round(centre, 3))


def polar_azimuth(p):
    d = p - centre
    r = np.linalg.norm(d, axis=-1)
    polar = np.degrees(np.arccos(np.clip(d[..., 2] / np.maximum(r, 1e-9), -1, 1)))
    azimuth = np.degrees(np.arctan2(d[..., 0], -d[..., 1]))  # 0 = facing forward (-Y)
    return polar, azimuth


def hairline(azimuth):
    """Largest polar angle (from the top) where hair grows: forehead, temples, above ears, nape."""
    a = np.abs(azimuth)
    return np.interp(a, [0, 25, 70, 110, 180], [FRONT_LIMIT, FRONT_LIMIT + 4, 92, 118, 128])


# ---------------------------------------------------------------- roots on the scalp
mesh = head.data
mesh.calc_loop_triangles()
tri = np.array([t.vertices[:] for t in mesh.loop_triangles])
co = np.array([head.matrix_world @ v.co for v in mesh.vertices])
a, b, c = co[tri[:, 0]], co[tri[:, 1]], co[tri[:, 2]]
area = np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2
mid = (a + b + c) / 3
polar, azimuth = polar_azimuth(mid)
inside = polar < hairline(azimuth)
if opt.get("HAIR_MASK"):
    # Where the head's own texture shows hair (project_texture.py's hair mask), hair grows too, so
    # the strands reach the hairline painted from the references and no bare band is left.
    image = bpy.data.images.load(opt["HAIR_MASK"])
    w, h = image.size
    px = np.empty(w * h * 4, np.float32)
    image.pixels.foreach_get(px)
    mask = px.reshape(h, w, 4)[:, :, 0]
    uv_layer = mesh.uv_layers.active.data
    tri_uv = np.array([[uv_layer[l].uv[:] for l in t.loops] for t in mesh.loop_triangles]).mean(axis=1)
    painted = mask[np.clip((tri_uv[:, 1] * h).astype(int), 0, h - 1), np.clip((tri_uv[:, 0] * w).astype(int), 0, w - 1)]
    # Only the scalp: not the face, ears or neck, even if their texture is light.
    inside = inside | ((painted > 0.4) & (polar < hairline(azimuth) + 25))
weights = area * inside
pick = rng.choice(len(tri), size=STRANDS, p=weights / weights.sum())
u, v = rng.random(STRANDS), rng.random(STRANDS)
flip = u + v > 1
u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
roots = a[pick] + (b[pick] - a[pick]) * u[:, None] + (c[pick] - a[pick]) * v[:, None]
print("ROOTS", STRANDS, "scalp triangles with hair", int(inside.sum()), "of", len(tri))


def hair_end(azimuth):
    """Polar angle where the hair stops: above the ear lobes at the sides, at the nape behind."""
    return np.interp(abs(azimuth), [0, 60, 90, 120, 180], [END_FRONT, END_FRONT + 25, END_SIDE, END_SIDE + 10, END_BACK])


def shell_radius(direction):
    hit = shell.ray_cast(Vector(centre), Vector(direction), 1.0)
    return (hit[0] - Vector(centre)).length if hit[0] is not None else None


def comb(n):
    """Combing direction at a point whose outward direction is n: back over the top, down the sides."""
    back = np.clip(n[2] * 1.6, 0, 1)
    v = np.array((0.0, 1.0, 0.0)) * back + np.array((0.0, 0.0, -1.0)) * (1.0 - 0.55 * back)
    # Over the temples hair sweeps back and away from the face rather than straight down.
    v += np.array((np.sign(n[0]) * 0.25 * abs(n[0]), 0.35 * max(-n[1], 0), 0.0))
    # Over the forehead, hair rises before it turns back, giving the front its height.
    v += np.array((0.0, 0.0, FRONT_RISE * max(-n[1] - 0.15, 0)))
    v = v - np.dot(v, n) * n
    return v / max(np.linalg.norm(v), 1e-9)


def grow(root, depth, phase, wave_scale):
    """Path of one guide strand: up into the volume, combed, waving, ending at the volume's edge."""
    n = (root - centre) / np.linalg.norm(root - centre)
    r_scalp = np.linalg.norm(root - centre)
    total = 0.0
    last_shell = None
    points = [root.copy()]
    step = 0.012
    for k in range(1, POINTS * 3):
        r_shell = shell_radius(n)
        p_now, a_now = polar_azimuth(centre + n)
        if r_shell is None:
            # Below the volume's edge, keep the last thickness, so hair can reach the hair_end line.
            r_shell = last_shell
        if r_shell is None or total > MAX_LENGTH or p_now > hair_end(a_now):
            break  # past the edge of the hairstyle
        last_shell = r_shell
        lift = min(total / LIFT, 1.0)
        lift = lift * lift * (3 - 2 * lift)
        r = r_scalp + (max(r_shell, r_scalp + 0.004) - r_scalp) * depth * lift
        direction = comb(n)
        side = np.cross(direction, n)
        wave = WAVE * wave_scale * math.sin(2 * math.pi * total / WAVELENGTH + phase)
        bulge = 0.35 * WAVE * wave_scale * math.cos(2 * math.pi * total / WAVELENGTH + phase)
        position = centre + n * (r + bulge * lift) + side * wave * lift
        points.append(position)
        n = n + direction * step / max(r, 1e-6)
        n = n / np.linalg.norm(n)
        total += step
    points = np.array(points)
    if len(points) < 4:
        return None
    # Roll the ends under, back toward the head, as set and rolled hair does (TIP_CURL_MM radius).
    tip = max(int(len(points) * TIP_SHARE), 2)
    start = points[-tip - 1].copy()
    out_dir = (start - centre) / np.linalg.norm(start - centre)
    travel = points[-1] - start
    travel -= np.dot(travel, out_dir) * out_dir
    travel /= max(np.linalg.norm(travel), 1e-9)
    for j in range(1, tip + 1):
        angle = math.pi * 0.9 * j / tip
        points[-tip - 1 + j] = start + travel * TIP_CURL * math.sin(angle) + out_dir * TIP_CURL * (1 - math.cos(angle)) * 0.6 * (1 if j < tip else 1)
    # Resample to a fixed number of points, evenly along its length.
    seg = np.linalg.norm(np.diff(points, axis=0), axis=1)
    s = np.concatenate([[0], np.cumsum(seg)])
    even = np.linspace(0, s[-1], POINTS)
    return np.stack([np.interp(even, s, points[:, i]) for i in range(3)], axis=1)


# ---------------------------------------------------------------- guides, then strands around them
guide_ids = rng.choice(STRANDS, size=GUIDES, replace=False)
guides = {}
for g in guide_ids:
    path = grow(roots[g], rng.uniform(0.75, 1.0), rng.uniform(0, 2 * math.pi), rng.uniform(0.6, 1.2))
    if path is not None:
        guides[g] = path
keys = np.array(list(guides))
print("GUIDES grown", len(keys))
guide_roots = roots[keys]
paths = np.zeros((STRANDS, POINTS, 3), np.float32)
along = np.linspace(0, 1, POINTS)[:, None]
for i in range(STRANDS):
    nearest = keys[np.argmin(np.linalg.norm(guide_roots - roots[i], axis=1))]
    g = guides[nearest]
    offset = roots[i] - g[0]
    # Children start at their own root and draw into their lock toward the tip, with a little
    # stray of their own so a lock isn't one solid ribbon.
    stray = rng.normal(0, STRAY, (POINTS, 3)) * along
    # Each strand also sits at its own depth within the lock, so a lock is round and full, not a
    # flat ribbon, and the layers underneath hide the scalp.
    outward = (g - centre) / np.linalg.norm(g - centre, axis=1, keepdims=True)
    depth = rng.uniform(-LAYER, 0.25 * LAYER)
    rise = np.clip(along * 4, 0, 1)
    paths[i] = g + offset * (1 - CLUMP * along) + outward * depth * rise + np.cumsum(stray, axis=0) * 0.3
lengths = np.linalg.norm(np.diff(paths, axis=1), axis=2).sum(axis=1)
print("STRAND_LENGTH_MM median", round(float(np.median(lengths)) * 1000), "range", round(float(lengths.min()) * 1000), "-", round(float(lengths.max()) * 1000))

# ---------------------------------------------------------------- the hair object
curves = bpy.data.hair_curves.new("hair_strands")
curves.add_curves([POINTS] * STRANDS)
curves.position_data.foreach_set("vector", paths.reshape(-1))
radius = curves.attributes.get("radius") or curves.attributes.new("radius", "FLOAT", "POINT")
taper = (ROOT_RADIUS * (1 - 0.65 * np.linspace(0, 1, POINTS) ** 1.5)).astype(np.float32)
radius.data.foreach_set("value", np.tile(taper, STRANDS))
obj = bpy.data.objects.new("hair_strands", curves)
bpy.context.scene.collection.objects.link(obj)

mat = bpy.data.materials.new("hair_strands")
mat.use_nodes = True
nodes, links = mat.node_tree.nodes, mat.node_tree.links
nodes.remove(nodes["Principled BSDF"])
hair = nodes.new("ShaderNodeBsdfHairPrincipled")
hair.parametrization = "MELANIN"
hair.inputs["Melanin"].default_value = MELANIN
hair.inputs["Melanin Redness"].default_value = float(opt.get("REDNESS", 0.2))
hair.inputs["Roughness"].default_value = float(opt.get("ROUGHNESS", 0.32))
hair.inputs["Radial Roughness"].default_value = 0.55
hair.inputs["Coat"].default_value = 0.15
hair.inputs["Random Color"].default_value = float(opt.get("RANDOM_COLOR", 0.12))
hair.inputs["Random Roughness"].default_value = 0.15
links.new(hair.outputs[0], nodes["Material Output"].inputs["Surface"])
curves.materials.append(mat)

rig = bpy.data.objects.get("joyce_rig")
if rig is not None and "head" in rig.data.bones:
    obj.parent = rig
    obj.parent_type = "BONE"
    obj.parent_bone = "head"
    bone = rig.data.bones["head"]
    bone_tail_world = rig.matrix_world @ bone.matrix_local @ Matrix.Translation((0, bone.length, 0))
    obj.matrix_parent_inverse = bone_tail_world.inverted()
for name in REMOVE:
    if name in bpy.data.objects:
        bpy.data.objects.remove(bpy.data.objects[name], do_unlink=True)
bpy.ops.wm.save_as_mainfile(filepath=out)
print("GROOM_OK", STRANDS, "strands of", POINTS, "points")
